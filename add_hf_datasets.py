"""
Augment an existing local HF dataset (created by preprocess_dataset.py) with
one or more Hugging Face datasets.

Each HF dataset can have a different schema — use the per-dataset flags to
configure text column, speaker column, and a prefix that namespaces speaker IDs
so they never collide across datasets.

Supported HF dataset schemas (examples):
  gsyllas/commonVoice_greek_clean_speakers_genders
    columns: file_name, transcription, transcription_normalised, audio, speaker_id, client_id, gender
  gsyllas/cs10_greek_dataset
    columns: file_name, transcription, transcription_normalised, audio, speaker_id, gender
  gsyllas/greek_male_3.5h
    columns: audio, text_whisper, text, gender, speaker_id, filename

Per-dataset configuration
--------------------------
  --per_dataset_text_column COL [COL ...]
      One text column per --hf_dataset entry. Falls back to --text_column.
  --per_dataset_speaker_column COL [COL ...]
      One speaker column per --hf_dataset entry. Falls back to --speaker_column.
  --speaker_id_prefix PREFIX [PREFIX ...]
      One prefix per --hf_dataset entry. Speaker names become "{prefix}_{original_id}".
      Essential when two datasets share the same speaker_id values.

Example — the three Greek datasets
-------------------------------------
python add_hf_datasets.py \\
    --dataset_dir ./my_dataset \\
    --hf_dataset gsyllas/commonVoice_greek_clean_speakers_genders \\
                 gsyllas/cs10_greek_dataset \\
                 gsyllas/greek_male_3.5h \\
    --per_dataset_text_column transcription_normalised transcription_normalised text \\
    --per_dataset_speaker_column speaker_id speaker_id speaker_id \\
    --speaker_id_prefix cv cs10 male3h \\
    --drop_english \\
    --output_dir ./my_dataset_augmented
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def contains_english(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def get_audio_duration(audio_value, enabled: bool = True) -> Optional[float]:
    """Return duration in seconds from an Audio feature dict, or None.

    Pass enabled=False to skip decoding (login-node safe mode).
    """
    if not enabled:
        return None
    try:
        arr = audio_value["array"]
        sr = audio_value["sampling_rate"]
        if sr and sr > 0 and arr is not None:
            return len(arr) / sr
    except Exception:
        pass
    return None


def load_existing_dataset(dataset_dir: str):
    obj = load_from_disk(dataset_dir)
    if isinstance(obj, DatasetDict):
        return obj, list(obj.keys())
    return obj, None


def load_speaker_mapping(dataset_dir: str) -> Dict[str, int]:
    path = os.path.join(dataset_dir, "speaker_id_mapping.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return dict(data.get("mapping", {}))


def save_speaker_mapping(output_dir: str, mapping: Dict[str, int]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "speaker_id_mapping.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"num_speakers": len(mapping), "mapping": mapping},
            fh,
            ensure_ascii=False,
            indent=2,
        )


def get_or_create_speaker_id(name: str, mapping: Dict[str, int]) -> int:
    if name not in mapping:
        mapping[name] = len(mapping)
    return mapping[name]


# ---------------------------------------------------------------------------
# Per-dataset config resolution
# ---------------------------------------------------------------------------

def _resolve_per_dataset(flag_values: Optional[List[str]], fallback: Optional[str], n: int) -> List[Optional[str]]:
    """Return a list of length n, picking from flag_values or falling back."""
    if not flag_values:
        return [fallback] * n
    if len(flag_values) == 1:
        return flag_values * n
    if len(flag_values) == n:
        return flag_values
    raise ValueError(
        f"Expected 1 or {n} values but got {len(flag_values)}."
    )


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

class StatsAccumulator:
    def __init__(self) -> None:
        # source -> list of (duration_sec|None, text_len, speaker_name|None)
        self._samples: Dict[str, List[Tuple[Optional[float], int, Optional[str]]]] = defaultdict(list)

    def add(self, source: str, duration: Optional[float], text_len: int, speaker_name: Optional[str]) -> None:
        self._samples[source].append((duration, text_len, speaker_name))

    def collect_from_dataset(self, dataset: Dataset, source: str, audio_stats: bool = True) -> None:
        has_speaker = "speaker_name" in dataset.column_names
        note = "" if audio_stats else " (audio decoding skipped)"
        print(f"  Collecting stats for '{source}' ({len(dataset):,} samples){note} …")
        for row in dataset:
            dur = get_audio_duration(row["audio"], enabled=audio_stats)
            text_len = len((row.get("text") or "").strip())
            spk = row.get("speaker_name") if has_speaker else None
            self.add(source, dur, text_len, spk)

    def build(self, speaker_mapping: Dict[str, int]) -> dict:
        # per-source
        per_source: Dict[str, dict] = {}
        for source, samples in self._samples.items():
            durations = [d for d, _, _ in samples if d is not None]
            text_lens = [t for _, t, _ in samples]
            per_source[source] = _agg_stats(samples, durations, text_lens)

        # per-speaker
        per_speaker_raw: Dict[str, Dict] = defaultdict(
            lambda: {"durations": [], "text_lens": [], "sources": set()}
        )
        for source, samples in self._samples.items():
            for dur, text_len, spk in samples:
                key = spk if spk is not None else "__no_speaker__"
                per_speaker_raw[key]["durations"].append(dur)
                per_speaker_raw[key]["text_lens"].append(text_len)
                per_speaker_raw[key]["sources"].add(source)

        per_speaker: Dict[str, dict] = {}
        for spk_name, data in sorted(per_speaker_raw.items(), key=lambda x: speaker_mapping.get(x[0], 9999)):
            durations = [d for d in data["durations"] if d is not None]
            samples_raw = [(d, t, None) for d, t in zip(data["durations"], data["text_lens"])]
            entry = _agg_stats(samples_raw, durations, data["text_lens"])
            entry["speaker_id"] = speaker_mapping.get(spk_name)
            entry["sources"] = sorted(data["sources"])
            per_speaker[spk_name] = entry

        # overall
        all_samples = [s for sl in self._samples.values() for s in sl]
        all_durations = [d for d, _, _ in all_samples if d is not None]
        all_text_lens = [t for _, t, _ in all_samples]
        overall = _agg_stats(all_samples, all_durations, all_text_lens)
        overall["num_speakers"] = len([k for k in per_speaker if k != "__no_speaker__"])

        return {"overall": overall, "per_source": per_source, "per_speaker": per_speaker}


def _agg_stats(samples, durations, text_lens) -> dict:
    return {
        "samples": len(samples),
        "duration_hours": round(sum(durations) / 3600, 4) if durations else None,
        "duration_seconds": round(sum(durations), 2) if durations else None,
        "avg_duration_seconds": round(sum(durations) / len(durations), 3) if durations else None,
        "min_duration_seconds": round(min(durations), 3) if durations else None,
        "max_duration_seconds": round(max(durations), 3) if durations else None,
        "avg_text_chars": round(sum(text_lens) / len(text_lens), 1) if text_lens else None,
        "min_text_chars": min(text_lens) if text_lens else None,
        "max_text_chars": max(text_lens) if text_lens else None,
    }


# ---------------------------------------------------------------------------
# HF dataset processing
# ---------------------------------------------------------------------------

def process_hf_dataset(
    repo_id: str,
    split: str,
    text_column: str,
    speaker_column: Optional[str],
    speaker_from_filename: bool,
    fixed_speaker_name: Optional[str],
    speaker_id_prefix: Optional[str],
    speaker_mapping: Dict[str, int],
    drop_english: bool,
    need_speaker: bool,
    stats: StatsAccumulator,
    audio_stats: bool = True,
) -> Dataset:
    print(f"\nDownloading {repo_id} (split={split}) …")
    hf_ds = load_dataset(repo_id, split=split, trust_remote_code=True)

    available = hf_ds.column_names
    if text_column not in available:
        raise ValueError(
            f"Text column '{text_column}' not found in {repo_id}.\n"
            f"Available columns: {available}\n"
            f"Hint: use --per_dataset_text_column to set the correct column per dataset."
        )
    if "audio" not in available:
        raise ValueError(f"'audio' column not found in {repo_id}. Available: {available}")
    if speaker_column and speaker_column not in available:
        raise ValueError(
            f"Speaker column '{speaker_column}' not found in {repo_id}.\n"
            f"Available columns: {available}\n"
            f"Hint: use --per_dataset_speaker_column to set the correct column per dataset."
        )

    # Use the filename column (could be "file_name" or "filename")
    filename_col = "file_name" if "file_name" in available else ("filename" if "filename" in available else None)

    filtered_english = 0
    skipped_speaker = 0
    records = []

    for row in hf_ds:
        text = (row[text_column] or "").strip()
        if not text:
            continue
        if drop_english and contains_english(text):
            filtered_english += 1
            continue

        record = {"audio": row["audio"], "text": text}

        speaker_name: Optional[str] = None
        if need_speaker:
            if speaker_column:
                raw = str(row[speaker_column]).strip()
                speaker_name = f"{speaker_id_prefix}_{raw}" if speaker_id_prefix else raw
            elif speaker_from_filename and filename_col:
                fname = (row.get(filename_col) or "").strip()
                parent = os.path.basename(os.path.dirname(fname))
                raw = parent if parent else None
                if raw:
                    speaker_name = f"{speaker_id_prefix}_{raw}" if speaker_id_prefix else raw
            elif fixed_speaker_name:
                speaker_name = f"{speaker_id_prefix}_{fixed_speaker_name}" if speaker_id_prefix else fixed_speaker_name

            if speaker_name is None or speaker_name == "":
                skipped_speaker += 1
                continue

            record["speaker_id"] = get_or_create_speaker_id(speaker_name, speaker_mapping)
            record["speaker_name"] = speaker_name

        dur = get_audio_duration(row["audio"], enabled=audio_stats)
        stats.add(repo_id, dur, len(text), speaker_name)
        records.append(record)

    print(f"  Kept {len(records):,} samples from {repo_id}")
    if filtered_english:
        print(f"  Filtered {filtered_english:,} samples with English characters")
    if skipped_speaker:
        print(f"  Skipped {skipped_speaker:,} samples with missing speaker")

    if not records:
        raise ValueError(f"No valid records from {repo_id} after filtering.")

    keys = records[0].keys()
    columns = {k: [r[k] for r in records] for k in keys}
    ds = Dataset.from_dict(columns)
    ds = ds.cast_column("audio", Audio())
    return ds


# ---------------------------------------------------------------------------
# Existing dataset helpers
# ---------------------------------------------------------------------------

def add_speaker_to_existing(dataset: Dataset, speaker_name: str, mapping: Dict[str, int]) -> Dataset:
    sid = get_or_create_speaker_id(speaker_name, mapping)
    n = len(dataset)
    return dataset.add_column("speaker_id", [sid] * n).add_column("speaker_name", [speaker_name] * n)


# ---------------------------------------------------------------------------
# Stats output
# ---------------------------------------------------------------------------

def _fmt_hours(h: Optional[float]) -> str:
    if h is None:
        return "N/A"
    total_sec = int(round(h * 3600))
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{hh}h {mm:02d}m {ss:02d}s"


def _fmt_sec(s: Optional[float]) -> str:
    return "N/A" if s is None else f"{s:.2f}s"


def save_stats(output_dir: str, stats_data: dict, dataset_dir: str, hf_datasets: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    full = {
        "generated_at": timestamp,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "hf_datasets_added": hf_datasets,
        **stats_data,
    }

    json_path = os.path.join(output_dir, "dataset_stats.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(full, fh, ensure_ascii=False, indent=2)

    # ---- human-readable report ----------------------------------------
    lines = []
    W = 72
    lines.append("=" * W)
    lines.append("DATASET STATISTICS")
    lines.append(f"Generated : {timestamp}")
    lines.append(f"Source    : {dataset_dir}")
    lines.append(f"Output    : {output_dir}")
    for ds in hf_datasets:
        lines.append(f"HF added  : {ds}")
    lines.append("=" * W)

    ov = stats_data["overall"]
    lines.append("")
    lines.append("OVERALL")
    lines.append(f"  Total samples          : {ov['samples']:,}")
    lines.append(f"  Total duration         : {_fmt_hours(ov['duration_hours'])}  ({ov['duration_hours']} h)")
    lines.append(f"  Number of speakers     : {ov.get('num_speakers', 'N/A')}")
    lines.append(f"  Avg duration / sample  : {_fmt_sec(ov['avg_duration_seconds'])}")
    lines.append(f"  Min / Max duration     : {_fmt_sec(ov['min_duration_seconds'])} / {_fmt_sec(ov['max_duration_seconds'])}")
    lines.append(f"  Avg text length (chars): {ov['avg_text_chars']}")
    lines.append(f"  Min / Max text chars   : {ov['min_text_chars']} / {ov['max_text_chars']}")

    lines.append("")
    lines.append("PER SOURCE")
    col_src = 42
    lines.append(f"  {'Source':<{col_src}} {'Samples':>8}  {'Duration':>14}  {'Avg dur':>8}  {'Avg chars':>9}")
    lines.append("  " + "-" * (col_src + 50))
    for src, sv in stats_data["per_source"].items():
        lines.append(
            f"  {src:<{col_src}} {sv['samples']:>8,}  "
            f"{_fmt_hours(sv['duration_hours']):>14}  "
            f"{_fmt_sec(sv['avg_duration_seconds']):>8}  "
            f"{str(sv['avg_text_chars'] or 'N/A'):>9}"
        )

    if stats_data["per_speaker"]:
        lines.append("")
        lines.append("PER SPEAKER")
        col_spk = 38
        lines.append(
            f"  {'ID':>4}  {'Speaker':<{col_spk}} {'Samples':>8}  "
            f"{'Duration':>14}  {'Avg dur':>8}  Sources"
        )
        lines.append("  " + "-" * (col_spk + 60))
        for spk_name, sv in stats_data["per_speaker"].items():
            sid = str(sv["speaker_id"]) if sv["speaker_id"] is not None else "-"
            src_str = ", ".join(sv.get("sources", []))
            lines.append(
                f"  {sid:>4}  {spk_name:<{col_spk}} {sv['samples']:>8,}  "
                f"{_fmt_hours(sv['duration_hours']):>14}  "
                f"{_fmt_sec(sv['avg_duration_seconds']):>8}  {src_str}"
            )

    lines.append("")
    lines.append("=" * W)

    txt_path = os.path.join(output_dir, "dataset_stats.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nStats saved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else dataset_dir
    n = len(args.hf_dataset)

    stats = StatsAccumulator()

    # ---- Resolve per-dataset config -----------------------------------
    try:
        per_text_cols = _resolve_per_dataset(args.per_dataset_text_column, args.text_column, n)
        per_spk_cols = _resolve_per_dataset(args.per_dataset_speaker_column, args.speaker_column, n)
        per_prefixes = _resolve_per_dataset(args.speaker_id_prefix, None, n)
        fixed_speaker_names = _resolve_per_dataset(args.speaker_name, None, n)
    except ValueError as exc:
        raise ValueError(f"Per-dataset argument length mismatch: {exc}") from exc

    # ---- Load existing dataset ----------------------------------------
    print(f"Loading existing dataset from {dataset_dir} …")
    existing_obj, split_names = load_existing_dataset(dataset_dir)
    speaker_mapping = load_speaker_mapping(dataset_dir)
    print(f"  Splits: {split_names or 'single (no split)'}")
    print(f"  Existing speakers: {len(speaker_mapping)}")

    ref_split = existing_obj[list(existing_obj.keys())[0]] if isinstance(existing_obj, DatasetDict) else existing_obj
    existing_has_speaker = "speaker_id" in ref_split.column_names

    # Do we need speaker info at all?
    need_speaker = (
        existing_has_speaker
        or any(c is not None for c in per_spk_cols)
        or args.speaker_from_filename
        or any(n_ is not None for n_ in fixed_speaker_names)
    )

    # Retrofit speaker info onto existing data if needed
    if not existing_has_speaker and need_speaker:
        name = args.existing_speaker_name
        print(f"\nExisting dataset has no speaker info — assigning '{name}' to all existing samples.")
        if isinstance(existing_obj, DatasetDict):
            existing_obj = DatasetDict(
                {s: add_speaker_to_existing(ds, name, speaker_mapping) for s, ds in existing_obj.items()}
            )
        else:
            existing_obj = add_speaker_to_existing(existing_obj, name, speaker_mapping)

    # ---- Collect stats for existing dataset ---------------------------
    audio_stats = not args.no_audio_stats
    print("\nCollecting stats for existing dataset …")
    if isinstance(existing_obj, DatasetDict):
        for split_name, ds in existing_obj.items():
            stats.collect_from_dataset(ds, f"existing:{split_name}", audio_stats=audio_stats)
    else:
        stats.collect_from_dataset(existing_obj, "existing", audio_stats=audio_stats)

    # ---- Download and process each HF dataset -------------------------
    new_datasets: List[Dataset] = []
    for i, repo_id in enumerate(args.hf_dataset):
        ds = process_hf_dataset(
            repo_id=repo_id,
            split=args.hf_split,
            text_column=per_text_cols[i],
            speaker_column=per_spk_cols[i],
            speaker_from_filename=args.speaker_from_filename,
            fixed_speaker_name=fixed_speaker_names[i],
            speaker_id_prefix=per_prefixes[i],
            speaker_mapping=speaker_mapping,
            drop_english=args.drop_english,
            need_speaker=need_speaker,
            stats=stats,
            audio_stats=audio_stats,
        )
        new_datasets.append(ds)

    # ---- Merge (train split only) -------------------------------------
    train_existing = existing_obj["train"] if isinstance(existing_obj, DatasetDict) else existing_obj
    merged_train = concatenate_datasets([train_existing] + new_datasets)
    print(f"\nMerged train size: {len(merged_train):,}")

    if isinstance(existing_obj, DatasetDict):
        result = DatasetDict({**existing_obj, "train": merged_train})
    else:
        result = merged_train

    # ---- Save dataset -------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving merged dataset to {output_dir} …")
    result.save_to_disk(output_dir) if isinstance(result, DatasetDict) else result.save_to_disk(output_dir)

    if speaker_mapping:
        save_speaker_mapping(output_dir, speaker_mapping)
        print(f"Updated speaker mapping: {len(speaker_mapping)} speakers")

    # ---- Stats --------------------------------------------------------
    stats_data = stats.build(speaker_mapping)
    save_stats(output_dir, stats_data, dataset_dir, args.hf_dataset)

    # ---- Terminal summary ---------------------------------------------
    ov = stats_data["overall"]
    total = sum(len(result[s]) for s in result) if isinstance(result, DatasetDict) else len(result)
    print(f"\nDone. Total samples in output : {total:,}")
    print(f"Total duration                : {_fmt_hours(ov['duration_hours'])}  ({ov['duration_hours']} h)")
    print(f"Speakers                      : {ov.get('num_speakers', 'N/A')}")

    if stats_data["per_speaker"]:
        print("\nPer-speaker summary:")
        for spk, sv in stats_data["per_speaker"].items():
            print(f"  [{sv['speaker_id']:>3}] {spk:<40} {sv['samples']:>6,} samples  {_fmt_hours(sv['duration_hours'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment an existing local HF dataset with Hugging Face datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Existing dataset ---
    parser.add_argument("--dataset_dir", required=True,
        help="Existing dataset folder (output of preprocess_dataset.py).")
    parser.add_argument("--output_dir", default=None,
        help="Where to save the merged dataset (default: in-place update of --dataset_dir).")

    # --- HF datasets ---
    parser.add_argument("--hf_dataset", required=True, nargs="+", metavar="REPO_ID",
        help="One or more HF dataset repository IDs to download and merge.")
    parser.add_argument("--hf_split", default="train",
        help="Split to download from each HF dataset (default: train).")

    # --- Global column defaults ---
    parser.add_argument("--text_column", default="transcription",
        help="Default text column name (default: transcription).")

    # --- Per-dataset column overrides ---
    parser.add_argument("--per_dataset_text_column", nargs="+", metavar="COL", default=None,
        help="Text column, one per --hf_dataset. Overrides --text_column per dataset. "
             "E.g. transcription_normalised transcription_normalised text")
    parser.add_argument("--per_dataset_speaker_column", nargs="+", metavar="COL", default=None,
        help="Speaker column, one per --hf_dataset. "
             "E.g. speaker_id speaker_id speaker_id")
    parser.add_argument("--speaker_id_prefix", nargs="+", metavar="PREFIX", default=None,
        help="Prefix added to speaker IDs to prevent cross-dataset collisions. "
             "One per --hf_dataset, or one for all. E.g. cv cs10 male3h")

    # --- Global speaker assignment fallbacks ---
    spk_group = parser.add_mutually_exclusive_group()
    spk_group.add_argument("--speaker_column", default=None, metavar="COL",
        help="Global fallback speaker column (used when --per_dataset_speaker_column is not set).")
    spk_group.add_argument("--speaker_from_filename", action="store_true",
        help="Infer speaker from parent dir of the filename column (global, all datasets).")
    spk_group.add_argument("--speaker_name", nargs="+", metavar="NAME",
        help="Fixed speaker name(s). One for all datasets, or one per --hf_dataset.")

    # --- Existing dataset speaker fallback ---
    parser.add_argument("--existing_speaker_name", default="existing_speaker", metavar="NAME",
        help="Speaker name for existing samples when the existing dataset has no speaker info "
             "(default: existing_speaker).")

    # --- Performance / login-node mode ---
    parser.add_argument("--no_audio_stats", action="store_true",
        help="Skip audio decoding during stats collection. "
             "Duration fields will be null in the stats file but the script runs "
             "with minimal CPU — safe for HPC login nodes. "
             "Sample counts and text stats are always computed.")

    # --- Filtering ---
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--drop_english", action="store_true",
        help="Filter out samples containing [a-zA-Z] in text.")
    filter_group.add_argument("--keep_english", action="store_true",
        help="Keep all text regardless of English characters.")

    parsed = parser.parse_args()

    # Default: filter English (mirrors preprocess_dataset.py)
    if not parsed.keep_english and not parsed.drop_english:
        parsed.drop_english = True

    main(parsed)

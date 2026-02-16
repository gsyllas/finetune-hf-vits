"""
Build a local Hugging Face dataset for VITS/MMS finetuning.

Default behavior matches the original single-speaker Greek preprocessing:
- reads `file_name|transcription` metadata with `|` delimiter
- resolves audio files from `audio_dir`
- filters out lines containing English characters
- saves a single-split `Dataset` (`audio`, `text`)

Optional flags enable multispeaker fields and train/eval splits.

Multi-CSV support:
- pass multiple `--csv_path` values
- pass one `--audio_dir` for all CSVs, or one `--audio_dir` per CSV
- when using `--speaker_column`, speaker IDs are derived from CSV values
  (e.g. `male`, `female`) and not from folder names
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from datasets import Audio, Dataset, DatasetDict


def contains_english(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))


def resolve_audio_path(path_value: str, audio_dirs: List[str]) -> Tuple[Optional[str], Optional[str], bool]:
    path_value = path_value.strip()
    candidates = []
    seen_candidates = set()

    def add_candidate(candidate_path: str) -> None:
        normalized_candidate = os.path.normpath(candidate_path)
        if normalized_candidate in seen_candidates:
            return
        seen_candidates.add(normalized_candidate)
        candidates.append(candidate_path)

    if os.path.isabs(path_value):
        add_candidate(path_value)
    for audio_dir in audio_dirs:
        add_candidate(os.path.join(audio_dir, path_value))
        add_candidate(os.path.join(audio_dir, os.path.basename(path_value)))

    existing_candidates = []
    seen_existing_candidates = set()
    for candidate in candidates:
        if os.path.exists(candidate):
            normalized_candidate = os.path.normpath(candidate)
            if normalized_candidate in seen_existing_candidates:
                continue
            seen_existing_candidates.add(normalized_candidate)
            existing_candidates.append(candidate)

    if not existing_candidates:
        return None, None, False

    selected_path = existing_candidates[0]
    matched_audio_dir = None
    for audio_dir in audio_dirs:
        normalized_audio_dir = os.path.normpath(audio_dir)
        normalized_selected_path = os.path.normpath(selected_path)
        if normalized_selected_path == normalized_audio_dir or normalized_selected_path.startswith(
            normalized_audio_dir + os.sep
        ):
            matched_audio_dir = audio_dir
            break

    return selected_path, matched_audio_dir, len(existing_candidates) > 1


def to_dataset(records: List[Dict[str, object]]) -> Dataset:
    if not records:
        raise ValueError("Cannot create dataset from an empty records list.")
    keys = records[0].keys()
    columns = {key: [record[key] for record in records] for key in keys}
    return Dataset.from_dict(columns).cast_column("audio", Audio())


def split_records(
    records: List[Dict[str, object]],
    eval_size: float,
    seed: int,
    has_speaker: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if eval_size <= 0:
        return records, []

    rng = random.Random(seed)
    indices = list(range(len(records)))
    train_indices = []
    eval_indices = []

    if not has_speaker:
        rng.shuffle(indices)
        eval_count = max(1, int(round(len(indices) * eval_size)))
        eval_count = min(eval_count, max(0, len(indices) - 1))
        eval_indices = indices[:eval_count]
        train_indices = indices[eval_count:]
    else:
        by_speaker = defaultdict(list)
        for idx, record in enumerate(records):
            by_speaker[record["speaker_id"]].append(idx)

        for speaker_indices in by_speaker.values():
            rng.shuffle(speaker_indices)
            if len(speaker_indices) == 1:
                train_indices.extend(speaker_indices)
                continue
            eval_count = max(1, int(round(len(speaker_indices) * eval_size)))
            eval_count = min(eval_count, len(speaker_indices) - 1)
            eval_indices.extend(speaker_indices[:eval_count])
            train_indices.extend(speaker_indices[eval_count:])

    train_indices = sorted(train_indices)
    eval_indices = sorted(eval_indices)

    train_records = [records[idx] for idx in train_indices]
    eval_records = [records[idx] for idx in eval_indices]
    return train_records, eval_records


def main(args: argparse.Namespace) -> None:
    csv_paths = [os.path.abspath(csv_path) for csv_path in args.csv_path]
    audio_dirs = [os.path.abspath(audio_dir) for audio_dir in args.audio_dir]

    if len(csv_paths) == 1:
        csv_audio_dirs = [audio_dirs]
    elif len(audio_dirs) == 1:
        csv_audio_dirs = [audio_dirs for _ in csv_paths]
    elif len(audio_dirs) == len(csv_paths):
        # Pair each CSV with its corresponding audio directory.
        csv_audio_dirs = [[audio_dirs[idx]] for idx in range(len(csv_paths))]
    else:
        raise ValueError(
            "Invalid --audio_dir / --csv_path combination. "
            "Use one --audio_dir for all CSVs, or provide one --audio_dir per --csv_path."
        )

    speaker_name_to_id: Dict[str, int] = {}
    records = []

    skipped_missing_audio = 0
    skipped_missing_text = 0
    skipped_missing_speaker = 0
    filtered_english = 0
    ambiguous_audio_matches = 0

    for csv_path, search_audio_dirs in zip(csv_paths, csv_audio_dirs):
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=args.delimiter)
            for row in reader:
                if args.path_column not in row:
                    raise ValueError(
                        f"Missing '{args.path_column}' column in metadata file '{csv_path}'. Available: {list(row.keys())}"
                    )
                if args.text_column not in row:
                    raise ValueError(
                        f"Missing '{args.text_column}' column in metadata file '{csv_path}'. Available: {list(row.keys())}"
                    )

                audio_path, matched_audio_dir, is_ambiguous_audio_match = resolve_audio_path(
                    row[args.path_column], search_audio_dirs
                )
                if audio_path is None:
                    skipped_missing_audio += 1
                    continue
                if is_ambiguous_audio_match:
                    ambiguous_audio_matches += 1

                text = row[args.text_column].strip()
                if not text:
                    skipped_missing_text += 1
                    continue

                if args.drop_english and contains_english(text):
                    filtered_english += 1
                    continue

                record = {"audio": audio_path, "text": text}

                speaker_name = None
                if args.speaker_column:
                    if args.speaker_column not in row:
                        raise ValueError(
                            f"Missing '{args.speaker_column}' column in metadata file '{csv_path}'. Available: {list(row.keys())}"
                        )
                    speaker_name = row[args.speaker_column].strip()
                elif args.speaker_from_audio_dir:
                    if matched_audio_dir is None:
                        raise ValueError(
                            "Could not infer speaker from audio directory. "
                            "Use --speaker_column or ensure files resolve under one of the provided --audio_dir paths."
                        )
                    speaker_name = os.path.basename(os.path.normpath(matched_audio_dir))
                elif args.speaker_from_parent_dir:
                    speaker_name = os.path.basename(os.path.dirname(audio_path))

                if speaker_name is not None:
                    if not speaker_name:
                        skipped_missing_speaker += 1
                        continue
                    speaker_id = speaker_name_to_id.setdefault(speaker_name, len(speaker_name_to_id))
                    record["speaker_id"] = speaker_id
                    record["speaker_name"] = speaker_name

                records.append(record)

    if not records:
        raise ValueError("No valid records found after preprocessing.")

    has_speaker = "speaker_id" in records[0]
    train_records, eval_records = split_records(
        records=records,
        eval_size=args.eval_size,
        seed=args.seed,
        has_speaker=has_speaker,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = to_dataset(train_records)
    if eval_records:
        eval_dataset = to_dataset(eval_records)
        dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
        dataset.save_to_disk(args.output_dir)
    else:
        train_dataset.save_to_disk(args.output_dir)

    if speaker_name_to_id:
        mapping_path = os.path.join(args.output_dir, "speaker_id_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "num_speakers": len(speaker_name_to_id),
                    "mapping": speaker_name_to_id,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    print(f"Found {len(records)} valid samples")
    print(f"Skipped {skipped_missing_audio} missing files")
    print(f"Filtered out {filtered_english} samples containing English")
    if ambiguous_audio_matches:
        print(
            "Warning: "
            f"{ambiguous_audio_matches} rows matched multiple audio files across --audio_dir entries; "
            "the first match was used."
        )
    if skipped_missing_text:
        print(f"Skipped {skipped_missing_text} samples with empty text")
    if skipped_missing_speaker:
        print(f"Skipped {skipped_missing_speaker} samples with empty speaker")
    print(f"Dataset saved to {args.output_dir}")
    print(f"Sample: {train_records[0]['text']}")
    if eval_records:
        print(f"Split sizes -> train: {len(train_records)}, eval: {len(eval_records)}")
    if speaker_name_to_id:
        print(f"Detected speakers: {len(speaker_name_to_id)}")
        first_speaker = next(iter(speaker_name_to_id.items()))
        print(f"Example speaker mapping: {first_speaker[0]} -> {first_speaker[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        required=True,
        nargs="+",
        help="One or more metadata CSV files.",
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        nargs="+",
        help="One or more base directories for audio files.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory where HF dataset is saved.")
    parser.add_argument("--delimiter", default="|", help="Metadata delimiter.")
    parser.add_argument("--path_column", default="file_name", help="Column that points to audio file path.")
    parser.add_argument("--text_column", default="transcription", help="Column that contains transcription text.")
    parser.add_argument("--speaker_column", default=None, help="Column that identifies speaker name/id.")
    parser.add_argument(
        "--speaker_from_audio_dir",
        action="store_true",
        help="Infer speaker name from the matched --audio_dir folder name.",
    )
    parser.add_argument(
        "--speaker_from_parent_dir",
        action="store_true",
        help="Infer speaker name from parent directory of each resolved audio file.",
    )
    parser.add_argument("--eval_size", type=float, default=0.0, help="Fraction of samples used for eval split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split.")
    parser.add_argument(
        "--drop_english",
        action="store_true",
        help="Filter out samples that contain [a-zA-Z] characters in text.",
    )
    parser.add_argument(
        "--keep_english",
        action="store_true",
        help="Disable English filtering and keep all text lines.",
    )

    parsed_args = parser.parse_args()
    if parsed_args.eval_size < 0 or parsed_args.eval_size >= 1:
        raise ValueError("--eval_size must be in [0, 1).")
    speaker_mode_count = sum(
        [bool(parsed_args.speaker_column), parsed_args.speaker_from_parent_dir, parsed_args.speaker_from_audio_dir]
    )
    if speaker_mode_count > 1:
        raise ValueError(
            "Use only one speaker mode: --speaker_column, --speaker_from_parent_dir, or --speaker_from_audio_dir."
        )
    if parsed_args.keep_english and parsed_args.drop_english:
        raise ValueError("Use either --drop_english or --keep_english, not both.")
    if parsed_args.keep_english:
        parsed_args.drop_english = False
    elif not parsed_args.drop_english:
        # Backward compatibility: default behavior is to filter English samples.
        parsed_args.drop_english = True

    main(parsed_args)

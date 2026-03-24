"""
Build a local HF dataset from a prepare_tts_dataset.py output folder
for single-speaker female VITS finetuning.

Expected dataset layout:
    dataset_root/
      metadata.csv        — '|'-delimited, columns: filename, speaker_id,
                            transcription, origin_dataset, gender
      wavs/
        spk_0001__000001.wav
        spk_0001__000002.wav
        ...

The 'filename' column holds a relative path (e.g. wavs/spk_0001__000001.wav)
resolved against --dataset_root.

Normalization applied before saving:
    - Lowercase all text
    - Strip punctuation not in the tokenizer vocab (silently)
    - Keep: space, apostrophe, hyphen, underscore, digits, in-vocab Latin subset,
      Greek lowercase

--dry_run: audit WITHOUT saving anything.
    Reports:
        - Duration statistics (read from wav headers — no decoding)
        - Samples containing digits (numbers in numerical form)
        - Samples containing out-of-vocab non-punctuation characters

Usage:
    # Audit first
    python preprocess_female_dataset.py \\
        --dataset_root /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/gsyllas_dataset \\
        --output_dir   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_vits_gsyllas \\
        --dry_run

    # Build dataset once audit is clean
    python preprocess_female_dataset.py \\
        --dataset_root /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/gsyllas_dataset \\
        --output_dir   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_vits_gsyllas
"""

import argparse
import csv
import os
import re
import unicodedata
import wave
from collections import defaultdict

from datasets import Audio, Dataset

# ---------------------------------------------------------------------------
# Tokenizer vocabulary (mms-tts-ell-train)
# ---------------------------------------------------------------------------
VOCAB_CHARS: set[str] = set(
    " '-_"
    "0123456789"
    "abehikmnoptyxz"           # Latin subset present in vocab
    "â"
    "αβγδεζηθικλμνξοπρσςτυφχψω"
    "ΐάέήίϊϋόύώ"
)

DIGIT_RE = re.compile(r"\d")


def is_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def normalize(text: str) -> tuple[str, set[str]]:
    lowered = text.lower()
    result = []
    oov = set()
    for ch in lowered:
        if ch in VOCAB_CHARS:
            result.append(ch)
        elif is_punct(ch):
            pass
        else:
            oov.add(ch)
    return "".join(result).strip(), oov


def wav_duration(path: str) -> float:
    """Read duration in seconds from wav header without decoding audio."""
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def duration_stats(durations: list[float]) -> dict:
    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    total = sum(durations_sorted)

    def percentile(p: float) -> float:
        return durations_sorted[min(int(p / 100 * n), n - 1)]

    return {
        "count": n,
        "total_h": total / 3600,
        "total_min": total / 60,
        "mean_s": total / n,
        "min_s": durations_sorted[0],
        "max_s": durations_sorted[-1],
        "p10_s": percentile(10),
        "p25_s": percentile(25),
        "p50_s": percentile(50),
        "p75_s": percentile(75),
        "p90_s": percentile(90),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    csv_path = os.path.join(args.dataset_root, "metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found in {args.dataset_root}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    # -----------------------------------------------------------------------
    # DRY RUN
    # -----------------------------------------------------------------------
    if args.dry_run:
        digit_samples: list[dict] = []
        oov_samples: list[dict] = []
        skipped_missing = 0
        skipped_empty = 0
        valid_durations: list[float] = []

        print("Reading wav headers for duration stats (this may take a moment)...")
        for row in rows:
            wav_path = os.path.join(args.dataset_root, row["filename"].strip())
            original_text = row["transcription"].strip()

            if not os.path.exists(wav_path):
                skipped_missing += 1
                continue
            if not original_text:
                skipped_empty += 1
                continue

            duration = wav_duration(wav_path)
            valid_durations.append(duration)
            _, oov_chars = normalize(original_text)

            if DIGIT_RE.search(original_text):
                digit_samples.append({
                    "wav": os.path.basename(wav_path),
                    "text": original_text,
                    "duration": duration,
                    "digits": sorted(set(DIGIT_RE.findall(original_text))),
                })

            if oov_chars:
                oov_samples.append({
                    "wav": os.path.basename(wav_path),
                    "text": original_text,
                    "duration": duration,
                    "oov": oov_chars,
                })

        valid_total = len(valid_durations)

        print("=" * 70)
        print("DRY RUN REPORT")
        print("=" * 70)
        print(f"Total rows in CSV      : {len(rows)}")
        print(f"Skipped (missing wav)  : {skipped_missing}")
        print(f"Skipped (empty text)   : {skipped_empty}")
        print(f"Valid samples          : {valid_total}")
        print()

        if valid_durations:
            s = duration_stats(valid_durations)
            print("--- Duration statistics (valid samples) ---")
            print(f"  Total   : {s['total_h']:.2f} h  ({s['total_min']:.1f} min)")
            print(f"  Mean    : {s['mean_s']:.2f} s")
            print(f"  Min     : {s['min_s']:.2f} s")
            print(f"  Max     : {s['max_s']:.2f} s")
            print(f"  p10     : {s['p10_s']:.2f} s")
            print(f"  p25     : {s['p25_s']:.2f} s")
            print(f"  p50     : {s['p50_s']:.2f} s")
            print(f"  p75     : {s['p75_s']:.2f} s")
            print(f"  p90     : {s['p90_s']:.2f} s")
        print()

        print(f"--- Samples with digits (numbers in numerical form): {len(digit_samples)} ---")
        if digit_samples:
            print(f"  Combined duration : {sum(d['duration'] for d in digit_samples)/60:.1f} min")
            print("  First 10 examples:")
            for d in digit_samples[:10]:
                print(f"    [{d['wav']}]  ({d['duration']:.2f}s)")
                print(f"      text   : {d['text']}")
                print(f"      digits : {d['digits']}")
        print()

        oov_char_freq: dict[str, int] = defaultdict(int)
        for item in oov_samples:
            for ch in item["oov"]:
                oov_char_freq[ch] += 1

        print(f"--- Samples with out-of-vocab non-punctuation chars: {len(oov_samples)} ---")
        if oov_char_freq:
            print(f"  Combined duration : {sum(d['duration'] for d in oov_samples)/60:.1f} min")
            print("  OOV character frequencies:")
            for ch, count in sorted(oov_char_freq.items(), key=lambda x: -x[1]):
                codepoint = f"U+{ord(ch):04X}"
                name = unicodedata.name(ch, "unknown")
                print(f"    '{ch}'  ({codepoint}  {name})  —  {count} sample(s)")
            print("  First 10 examples:")
            for item in oov_samples[:10]:
                print(f"    [{item['wav']}]  ({item['duration']:.2f}s)")
                print(f"      text     : {item['text']}")
                print(f"      oov chars: {sorted(item['oov'])}")
        print()
        print("No dataset was written. Remove --dry_run to build the dataset.")
        return

    # -----------------------------------------------------------------------
    # NORMAL RUN
    # -----------------------------------------------------------------------
    audio_paths: list[str] = []
    texts: list[str] = []
    skipped_missing = 0
    skipped_empty = 0
    skipped_oov = 0

    for row in rows:
        wav_path = os.path.join(args.dataset_root, row["filename"].strip())
        original_text = row["transcription"].strip()

        if not os.path.exists(wav_path):
            skipped_missing += 1
            continue
        if not original_text:
            skipped_empty += 1
            continue

        normalized_text, oov_chars = normalize(original_text)
        if not normalized_text:
            skipped_empty += 1
            continue
        if oov_chars:
            skipped_oov += 1
            continue

        audio_paths.append(wav_path)
        texts.append(normalized_text)

    if not audio_paths:
        raise ValueError("No valid samples found after preprocessing.")

    dataset = Dataset.from_dict({"audio": audio_paths, "text": texts}).cast_column("audio", Audio())
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    print(f"Saved {len(audio_paths)} samples to {args.output_dir}")
    print(f"Skipped {skipped_missing} rows with missing wav file")
    print(f"Skipped {skipped_empty} rows with empty/blank transcription")
    print(f"Skipped {skipped_oov} rows with out-of-vocab non-punctuation characters")
    print(f"Sample: {texts[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Root of the prepare_tts_dataset.py output folder.")
    parser.add_argument("--output_dir", required=True, help="Where to save the HF dataset.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Audit only: print stats and OOV report. Does NOT write any dataset.",
    )
    main(parser.parse_args())

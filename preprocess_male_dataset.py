"""
Build a local HF dataset from a pipe-delimited CSV for single-speaker male VITS finetuning.

CSV format ('|' delimiter, with header):
    file_name | transcription | speaker_id | duration_seconds

The wav filename (basename of file_name) is resolved against --audio_dir.

Normalization applied before saving:
    - Lowercase all text
    - Strip punctuation not in the tokenizer vocab (silently)
    - Keep: space, apostrophe, hyphen, underscore, digits, in-vocab Latin subset, Greek lowercase

--dry_run: audit the dataset WITHOUT saving anything.
    Reports:
        - Duration statistics (total, mean, min, max, percentiles)
        - Samples containing digits (numbers in numerical form)
        - Samples containing out-of-vocab non-punctuation characters
          (characters from other languages/scripts)
    Remove --dry_run only once you are satisfied with the report.

Usage:
    # Audit first
    python preprocess_male_dataset.py \\
        --csv_path  /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male.csv \\
        --audio_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male \\
        --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_vits \\
        --dry_run

    # Build dataset once audit is clean
    python preprocess_male_dataset.py \\
        --csv_path  /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male.csv \\
        --audio_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male \\
        --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_vits
"""

import argparse
import csv
import os
import re
import unicodedata
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
    """True for punctuation and symbol Unicode categories — silently stripped."""
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def normalize(text: str) -> tuple[str, set[str]]:
    """
    Lowercase and strip punctuation.
    Returns (normalized_text, oov_non_punct_chars) where oov_non_punct_chars
    are characters that are neither in vocab nor classified as punctuation.
    """
    lowered = text.lower()
    result = []
    oov = set()
    for ch in lowered:
        if ch in VOCAB_CHARS:
            result.append(ch)
        elif is_punct(ch):
            pass  # strip silently
        else:
            oov.add(ch)
            # strip but flag
    return "".join(result).strip(), oov


def duration_stats(durations: list[float]) -> dict:
    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    total = sum(durations_sorted)
    mean = total / n
    def percentile(p: float) -> float:
        idx = int(p / 100 * n)
        return durations_sorted[min(idx, n - 1)]
    return {
        "count": n,
        "total_h": total / 3600,
        "total_min": total / 60,
        "mean_s": mean,
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
    rows = []
    with open(args.csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {args.csv_path}")

    # -----------------------------------------------------------------------
    # DRY RUN — audit only, no dataset written
    # -----------------------------------------------------------------------
    if args.dry_run:
        digit_samples: list[dict] = []
        oov_samples: list[dict] = []
        skipped_missing = 0
        skipped_empty = 0
        valid_durations: list[float] = []
        all_durations: list[float] = []

        for row in rows:
            wav_name = os.path.basename(row["file_name"].strip())
            wav_path = os.path.join(args.audio_dir, wav_name)
            original_text = row["transcription"].strip()
            try:
                duration = float(row["duration_seconds"].strip())
            except (ValueError, KeyError):
                duration = 0.0
            all_durations.append(duration)

            if not os.path.exists(wav_path):
                skipped_missing += 1
                continue
            if not original_text:
                skipped_empty += 1
                continue

            valid_durations.append(duration)
            _, oov_chars = normalize(original_text)

            if DIGIT_RE.search(original_text):
                digit_samples.append({
                    "wav": wav_name,
                    "text": original_text,
                    "duration": duration,
                    "digits": sorted(set(DIGIT_RE.findall(original_text))),
                })

            if oov_chars:
                oov_samples.append({
                    "wav": wav_name,
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

        # Duration stats on valid samples
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

        # Digit report
        print(f"--- Samples with digits (numbers in numerical form): {len(digit_samples)} ---")
        if digit_samples:
            digit_dur = sum(d["duration"] for d in digit_samples)
            print(f"  Combined duration : {digit_dur/60:.1f} min")
            print("  First 10 examples:")
            for d in digit_samples[:10]:
                print(f"    [{d['wav']}]  ({d['duration']:.2f}s)")
                print(f"      text   : {d['text']}")
                print(f"      digits : {d['digits']}")
        print()

        # OOV char report
        oov_char_freq: dict[str, int] = defaultdict(int)
        for item in oov_samples:
            for ch in item["oov"]:
                oov_char_freq[ch] += 1

        print(f"--- Samples with out-of-vocab non-punctuation chars: {len(oov_samples)} ---")
        if oov_char_freq:
            oov_dur = sum(d["duration"] for d in oov_samples)
            print(f"  Combined duration : {oov_dur/60:.1f} min")
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
    # NORMAL RUN — normalize and save dataset
    # -----------------------------------------------------------------------
    audio_paths: list[str] = []
    texts: list[str] = []
    skipped_missing = 0
    skipped_empty = 0

    for row in rows:
        wav_name = os.path.basename(row["file_name"].strip())
        wav_path = os.path.join(args.audio_dir, wav_name)
        original_text = row["transcription"].strip()

        if not os.path.exists(wav_path):
            skipped_missing += 1
            continue
        if not original_text:
            skipped_empty += 1
            continue

        normalized_text, _ = normalize(original_text)
        if not normalized_text:
            skipped_empty += 1
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
    print(f"Sample: {texts[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the pipe-delimited metadata CSV.")
    parser.add_argument("--audio_dir", required=True, help="Directory containing the .wav files.")
    parser.add_argument("--output_dir", required=True, help="Where to save the HF dataset.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Audit only: print duration stats, digit samples, and OOV char samples. "
            "Does NOT write any dataset to disk."
        ),
    )
    main(parser.parse_args())

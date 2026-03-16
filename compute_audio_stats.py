"""
Compute audio-duration statistics for an existing local HF dataset.

Run this after add_hf_datasets.py --no_audio_stats to fill in the
duration fields that were skipped on the login node.

Overwrites dataset_stats.json and dataset_stats.txt in --dataset_dir.

Example
-------
python compute_audio_stats.py --dataset_dir ./my_dataset_augmented
"""

import argparse
import json
import os
from datetime import datetime, timezone

from datasets import DatasetDict, load_from_disk

from add_hf_datasets import StatsAccumulator, load_speaker_mapping, save_stats


def main(args: argparse.Namespace) -> None:
    dataset_dir = os.path.abspath(args.dataset_dir)

    print(f"Loading dataset from {dataset_dir} …")
    obj = load_from_disk(dataset_dir)
    speaker_mapping = load_speaker_mapping(dataset_dir)

    splits = obj if isinstance(obj, DatasetDict) else {"train": obj}
    total = sum(len(ds) for ds in splits.values())
    print(f"  {len(splits)} split(s), {total:,} samples total")
    print(f"  {len(speaker_mapping)} speakers in mapping")

    # Load existing stats file to preserve hf_datasets_added metadata
    stats_path = os.path.join(dataset_dir, "dataset_stats.json")
    hf_datasets_added = []
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        hf_datasets_added = old.get("hf_datasets_added", [])

    stats = StatsAccumulator()
    for split_name, ds in splits.items():
        source = f"existing:{split_name}" if len(splits) > 1 else "existing"
        stats.collect_from_dataset(ds, source, audio_stats=True)

    stats_data = stats.build(speaker_mapping)
    save_stats(
        output_dir=dataset_dir,
        stats_data=stats_data,
        dataset_dir=dataset_dir,
        hf_datasets=hf_datasets_added,
    )

    ov = stats_data["overall"]
    from add_hf_datasets import _fmt_hours, _fmt_sec
    print(f"\nTotal samples   : {ov['samples']:,}")
    print(f"Total duration  : {_fmt_hours(ov['duration_hours'])}  ({ov['duration_hours']} h)")
    print(f"Avg / sample    : {_fmt_sec(ov['avg_duration_seconds'])}")
    if stats_data["per_speaker"]:
        print("\nPer-speaker:")
        for spk, sv in stats_data["per_speaker"].items():
            print(f"  [{sv['speaker_id']:>3}] {spk:<40} {sv['samples']:>6,} samples  {_fmt_hours(sv['duration_hours'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute audio stats for an existing local HF dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_dir", required=True,
        help="Path to the dataset folder to analyse.")
    main(parser.parse_args())

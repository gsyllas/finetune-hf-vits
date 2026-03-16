"""
Download and cache Hugging Face datasets locally.
Run this once before add_hf_datasets.py so the merge job can run offline.

Example
-------
python download_hf_datasets.py \
    --hf_dataset gsyllas/commonVoice_greek_clean_speakers_genders \
                 gsyllas/cs10_greek_dataset \
                 gsyllas/greek_male_3.5h \
    --hf_split train
"""

import argparse
import os

from datasets import load_dataset


def main(args: argparse.Namespace) -> None:
    print(f"HF cache dir: {os.environ.get('HF_DATASETS_CACHE', '~/.cache/huggingface/datasets')}")
    for repo_id in args.hf_dataset:
        print(f"\nDownloading {repo_id} (split={args.hf_split}) …")
        ds = load_dataset(repo_id, split=args.hf_split, trust_remote_code=True)
        print(f"  Done — {len(ds):,} examples, columns: {ds.column_names}")
    print("\nAll datasets cached. You can now run add_hf_datasets.py with HF_DATASETS_OFFLINE=1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download HF datasets to local cache.")
    parser.add_argument("--hf_dataset", required=True, nargs="+", metavar="REPO_ID")
    parser.add_argument("--hf_split", default="train")
    main(parser.parse_args())

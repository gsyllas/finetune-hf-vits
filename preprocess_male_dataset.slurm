#!/bin/bash
#SBATCH --job-name=preprocess-male
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

export HF_DATASETS_OFFLINE=1

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

# --- Audit first (dry run) ---
# Uncomment the block below to run the audit without saving anything.
# Check logs/%x_%j.out for the full report.
#
# python preprocess_male_dataset.py \
#     --csv_path  /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male.csv \
#     --audio_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male \
#     --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_vits \
#     --dry_run

# --- Build dataset (run only after dry_run is clean) ---
python preprocess_male_dataset.py \
    --csv_path  /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male.csv \
    --audio_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male \
    --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_vits

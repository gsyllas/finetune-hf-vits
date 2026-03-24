#!/bin/bash
#SBATCH --job-name=preprocess-female
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
# python preprocess_female_dataset.py \
#     --dataset_root /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/gsyllas_dataset \
#     --output_dir   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_vits_gsyllas \
#     --dry_run

# --- Build dataset (run only after dry_run is clean) ---
python preprocess_female_dataset.py \
    --dataset_root /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/gsyllas_dataset \
    --output_dir   /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_vits_gsyllas

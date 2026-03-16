#!/bin/bash
#SBATCH --job-name=audio-stats
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

python compute_audio_stats.py \
    --dataset_dir /leonardo_work/EUHPC_D29_081/gsyllas0/my_dataset_augmented

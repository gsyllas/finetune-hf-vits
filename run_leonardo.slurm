#!/bin/bash
#SBATCH --job-name=vits-greek
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module purge
module load cuda/12.2

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

# Set offline mode (no internet on compute)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Copy dataset and model to fast local storage
echo "Copying data to local SSD..."
cp -r /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/female_vits $TMPDIR/female_vits
cp -r /leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train $TMPDIR/mms-tts-ell-train
echo "Copy done."

# Go to project directory
cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

# Run training
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_greek.json


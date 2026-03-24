#!/bin/bash
#SBATCH --job-name=vits-greek-male
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module purge
module load cuda/12.2

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

# Checkpoint and output location
OUTPUT_DIR=/leonardo_work/EUHPC_D29_081/gsyllas0/output/single_speaker_male_mms_finetuning
mkdir -p $OUTPUT_DIR

# WandB: offline mode — sync manually after the run with:
#   wandb sync $OUTPUT_DIR/wandb/offline-run-${WANDB_RUN_ID}/
# Description: single-speaker MMS Greek male voice finetuning on Leonardo HPC
WANDB_RUN_ID="single_speaker_male_mms_finetuning_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
export WANDB_MODE=offline
export WANDB_PROJECT=single_speaker_male_mms_finetuning
export WANDB_NAME=$WANDB_RUN_ID
export WANDB_RUN_ID=$WANDB_RUN_ID
export WANDB_NOTES="Single-speaker MMS Greek male voice finetuning. SLURM job ${SLURM_JOB_ID} on Leonardo HPC (boost_usr_prod, 1xA100). Model: mms-tts-ell-train. Dataset: male_vits."
export WANDB_DIR=$OUTPUT_DIR

# Set offline mode (no internet on compute)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Copy dataset and model to fast local storage
echo "Copying data to local SSD..."
cp -r /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/male_vits $TMPDIR/male_vits
cp -r /leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train $TMPDIR/mms-tts-ell-train
echo "Copy done."

# Go to project directory
cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

# Run training
accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_greek_male.json

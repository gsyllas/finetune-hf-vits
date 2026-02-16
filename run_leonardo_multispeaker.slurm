#!/bin/bash
#SBATCH --job-name=vits-greek-multi
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load cuda/12.2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

accelerate launch run_vits_finetuning.py ./training_config_examples/finetune_greek_multispeaker.json

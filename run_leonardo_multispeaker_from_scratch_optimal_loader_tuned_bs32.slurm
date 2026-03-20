#!/bin/bash
#SBATCH --job-name=vits-greek-multi-opt-loader32
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits/logs/%x_%j.err

module purge
module load cuda/12.2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

accelerate launch run_vits_finetuning.py ./training_config_examples/from_scratch_greek_multispeaker_optimal_loader_tuned_bs32.json \
    --dataset_name /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined \
    --speaker_stats_path /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined/dataset_stats.json \
    --min_speaker_hours 3.0 \
    --config_name /leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train \
    --tokenizer_name /leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train \
    --feature_extractor_name /leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train \
    --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/output/vits_from_scratch_greek_multispeaker_optimal_loader_tuned_bs32 \
    --run_name greek-multi-scratch-optimal-loader-tuned-bs32 \
    --resume_from_checkpoint latest

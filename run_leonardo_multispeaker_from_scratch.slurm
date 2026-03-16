#!/bin/bash
#SBATCH --job-name=vits-greek-multi-scratch
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
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

# Override these at submission time if needed, e.g.
# sbatch --export=ALL,MIN_SPEAKER_HOURS=0.5,OUTPUT_DIR=/path/to/output run_leonardo_multispeaker_from_scratch.slurm
CONFIG_JSON=${CONFIG_JSON:-./training_config_examples/from_scratch_greek_multispeaker.json}
DATASET_DIR=${DATASET_DIR:-/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined}
SPEAKER_STATS_PATH=${SPEAKER_STATS_PATH:-${DATASET_DIR}/dataset_stats.json}
MIN_SPEAKER_HOURS=${MIN_SPEAKER_HOURS:-1.0}
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-/leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train}
OUTPUT_DIR=${OUTPUT_DIR:-/leonardo_work/EUHPC_D29_081/gsyllas0/output/vits_from_scratch_greek_multispeaker_combined_1h}

accelerate launch run_vits_finetuning.py "${CONFIG_JSON}" \
    --dataset_name "${DATASET_DIR}" \
    --speaker_stats_path "${SPEAKER_STATS_PATH}" \
    --min_speaker_hours "${MIN_SPEAKER_HOURS}" \
    --config_name "${ARTIFACT_SOURCE}" \
    --tokenizer_name "${ARTIFACT_SOURCE}" \
    --feature_extractor_name "${ARTIFACT_SOURCE}" \
    --output_dir "${OUTPUT_DIR}"

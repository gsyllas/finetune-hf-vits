#!/bin/bash
#SBATCH --job-name=vits-scratch-v2
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits/logs/%x_%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits/logs/%x_%j.err
#SBATCH --signal=B:USR1@600

# ── Signal handler ──────────────────────────────────────────────────
# SLURM sends USR1 600s (10 min) before the 24h wall-time.
# We touch the sentinel file so the training loop saves a checkpoint
# and exits cleanly, then requeue this same script.
handle_timeout() {
    echo "$(date): Caught USR1 — requesting clean stop + requeue"
    touch "${OUTPUT_DIR}/.slurm_resume_requested"
    # Wait for the training process to notice the file and exit
    wait
    echo "$(date): Training exited. Resubmitting..."
    sbatch "$0"
}
trap 'handle_timeout' USR1

# ── Environment ─────────────────────────────────────────────────────
module purge
module load cuda/12.2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

# ── Configuration ───────────────────────────────────────────────────
CONFIG_JSON=${CONFIG_JSON:-./training_config_examples/from_scratch_greek_multispeaker_v2.json}
DATASET_DIR=${DATASET_DIR:-/leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined}
SPEAKER_STATS_PATH=${SPEAKER_STATS_PATH:-${DATASET_DIR}/dataset_stats.json}
MIN_SPEAKER_HOURS=${MIN_SPEAKER_HOURS:-3.0}
ARTIFACT_SOURCE=${ARTIFACT_SOURCE:-/leonardo_work/EUHPC_D29_081/gsyllas0/models/mms-tts-ell-train}
OUTPUT_DIR=${OUTPUT_DIR:-/leonardo_work/EUHPC_D29_081/gsyllas0/output/vits_scratch_greek_ms_v2}
RUN_NAME=${RUN_NAME:-scratch-ms-v2-4gpu}
PROJECT_NAME=${PROJECT_NAME:-vits_greek_ms_scratch_v2}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-16}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-16}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
NUM_GPUS=${NUM_GPUS:-4}

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${RUN_NAME}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-$((10000 + SLURM_JOB_ID % 50000))}
else
    MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
fi

# ── Launch training (in background so trap can fire) ────────────────
accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    run_vits_finetuning.py "${CONFIG_JSON}" \
    --project_name "${PROJECT_NAME}" \
    --dataset_name "${DATASET_DIR}" \
    --speaker_stats_path "${SPEAKER_STATS_PATH}" \
    --min_speaker_hours "${MIN_SPEAKER_HOURS}" \
    --config_name "${ARTIFACT_SOURCE}" \
    --tokenizer_name "${ARTIFACT_SOURCE}" \
    --feature_extractor_name "${ARTIFACT_SOURCE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --resume_from_checkpoint latest &

# Wait for training; if USR1 arrives the trap fires *during* wait
wait $!
EXIT_CODE=$?

echo "$(date): Training finished with exit code ${EXIT_CODE}"

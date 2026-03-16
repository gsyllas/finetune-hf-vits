#!/bin/bash
#SBATCH --job-name=hf-merge
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

python add_hf_datasets.py \
    --dataset_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/greek_multispeaker_vits \
    --hf_dataset gsyllas/commonVoice_greek_clean_with_speaker_ids \
                 gsyllas/cs10_greek_dataset \
                 gsyllas/greek_male_3.5h \
    --per_dataset_text_column transcription_normalised transcription_normalised text \
    --per_dataset_speaker_column speaker_id speaker_id speaker_id \
    --speaker_id_prefix cv cs10 male3h \
    --drop_english \
    --no_audio_stats \
    --output_dir /leonardo_work/EUHPC_D29_081/gsyllas0/data/tts/multi_speaker_combined

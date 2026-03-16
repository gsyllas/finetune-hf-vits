#!/bin/bash
#SBATCH --job-name=hf-download
#SBATCH --account=EUHPC_D29_081
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vits

cd /leonardo_work/EUHPC_D29_081/gsyllas0/finetune-hf-vits

python download_hf_datasets.py \
    --hf_dataset gsyllas/commonVoice_greek_clean_speakers_genders \
                 gsyllas/cs10_greek_dataset \
                 gsyllas/greek_male_3.5h \
    --hf_split train

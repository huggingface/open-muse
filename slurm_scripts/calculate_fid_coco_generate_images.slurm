#!/bin/bash
#SBATCH --job-name=muse-calculate-fid-coco
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/calculate_fid_coco_generate_images.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/calculate_fid.py --slurm --model_name_or_path openMUSE/muse-laiona6-uvit-clip-220k --dataset_root ./data/fid_coco_30k_real --save_path ./data/fid_coco_30k_generated --batch_size 32 --do generate_and_save_images --dataset coco

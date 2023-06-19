#!/bin/bash
#SBATCH --job-name=muse-pre-encode-laion-5
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_pre_encode_laion_5.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/pre_encode.py --slurm --dataset laion_5 --start_shard 0 --end_shard 60580
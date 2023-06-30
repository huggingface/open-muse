#!/bin/bash
#SBATCH --job-name=muse-pre-encode-coyo
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_pre_encode_coyo.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/pre_encode.py --slurm --dataset coyo --start_shard 4235 --end_shard 74751

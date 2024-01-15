#!/bin/bash
#SBATCH --job-name=muse-gen-sdxl-synthetic-dataset
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_gen_sdxl_synthetic_dataset

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/gen_sdxl_synthetic_dataset.py --slurm --n_shards_to_write 400

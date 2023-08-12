#!/bin/bash
#SBATCH --job-name=muse-m4-annotate
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_m4_annotate.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/m4_annotate/m4_annotate.py --slurm --start_shard 0 --end_shard 71464

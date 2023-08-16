#!/bin/bash
#SBATCH --job-name=muse-stability-metadata-annotate
#SBATCH --nodes=15
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_stability_metadata_annotate.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/stability_metadata_annotate/stability_metadata_annotate.py --slurm --start_shard 0 --end_shard 71464

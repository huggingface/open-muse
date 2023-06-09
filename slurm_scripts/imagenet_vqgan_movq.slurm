#!/bin/bash
#SBATCH --job-name=vqgan_testing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -A cstdl
#SBATCH --partition booster
#SBATCH --output=/p/home/jusers/isozaki1/juwels/%x-%j.out
#SBATCH --time=0:06:00

set -x -e

echo "START TIME: $(date)"

MUSE_REPO=/p/home/jusers/isozaki1/juwels/open-muse
OUTPUT_DIR=/p/home/jusers/isozaki1/juwels/4_node_6_min
LOG_PATH=$OUTPUT_DIR/main_log.txt

mkdir -p $OUTPUT_DIR
touch $LOG_PATH
pushd $MUSE_REPO
source /p/home/jusers/isozaki1/juwels/miniconda3/etc/profile.d/conda.sh
conda activate muse
CMD=" \
    training/train_vqgan.py config=configs/imagenet_vqgan_training_jewels_f16_vqgan.yaml \
    wandb.entity=isamu \
    experiment.name=$(basename $OUTPUT_DIR) \
    experiment.output_dir=$OUTPUT_DIR \
    training.seed=9345104 \
    "

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

echo "END TIME: $(date)"

#!/bin/bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --qos=m2
#SBATCH --array=0

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

FOLD=$SLURM_ARRAY_TASK_ID
N_FOLDS=10
SPLITS_PATH=splits/ssl_fold${FOLD}:${N_FOLDS}.json
DATA_TYPE=rf
CHECKPOINT_PATH=/checkpoint/$USER/$SLURM_JOB_ID/checkpoint.pt

# Set environment variables for training
export TQDM_MININTERVAL=60

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

srun python train_patch_ssl_v2.py \
    --splits_file $SPLITS_PATH \
    --batch_size 64 \
    --lr 1e-4 \
    --data_type $DATA_TYPE \
    --name patch_ssl_${CENTER}_${DATA_TYPE}_${VAL_SEED} \
    --checkpoint_path=$CHECKPOINT_PATH \
    --save_weights_path=ssl_checkpoints/fold${FOLD}:${N_FOLDS}_rf_ssl_weights.pt \
    &

child_pid=$!
wait $child_pid

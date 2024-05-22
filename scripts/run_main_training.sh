#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --output=.slurm_logs/adapter_tuning_rf-%j.log
#SBATCH --open-mode=append
#SBATCH --qos=m2
#SBATCH --array=0

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

FOLD=$SLURM_ARRAY_TASK_ID
N_FOLDS=10
EXP_NAME=fold-${FOLD}_reprod
RUN_ID=$SLURM_JOB_ID
EXP_DIR=experiments/${EXP_NAME}/$RUN_ID
CKPT_DIR=/checkpoint/$USER/$RUN_ID
VAL_SEED=0

# Set environment variables for training
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$RUN_ID
export WANDB_RESUME=allow
export PYTHONUNBUFFERED=1

# Create experiment directory
echo "EXP_DIR: $EXP_DIR"
mkdir -p $EXP_DIR
# Symbolic link to checkpoint directory
# so it is easier to find them
echo "CKPT_DIR: $CKPT_DIR"
if [ ! -d $(realpath $EXP_DIR)/checkpoints ]; then
  ln -s $CKPT_DIR $(realpath $EXP_DIR)/checkpoints
fi
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

# Run training script
srun -u python train_prostnfound_refactor.py \
    --config_path conf/main_training.json \
    --exp_dir $EXP_DIR \
    --checkpoint_dir $CKPT_DIR \
    --splits_json_path splits/fold${FOLD}:${N_FOLDS}.json \
    --prompt_table_csv_path prompt_table.csv \
    --sparse_cnn_backbone_path ssl_checkpoints/fold${FOLD}:${N_FOLDS}_rf_ssl_weights.pt \
    --name $EXP_NAME &

child_pid=$!
wait $child_pid


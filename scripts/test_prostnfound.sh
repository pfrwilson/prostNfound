#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16
#SBATCH --output=%j.log
#SBATCH --open-mode=append
#SBATCH --qos=normal

# replace args with saved config, model weights, and output directory

srun python test_prostnfound.py \
    -c /fs01/home/pwilson/projects/prostNfound/experiments/fold-0_reprod/12676934/checkpoints/config.json \
    -m /fs01/home/pwilson/projects/prostNfound/experiments/fold-0_reprod/12676934/checkpoints/best_model.ckpt \
    -o .test

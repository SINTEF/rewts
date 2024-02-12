#!/bin/bash
#SBATCH --output "tb.out"
#SBATCH --job-name Tensorboard-TSA
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time 01-00:00:00

cd logs/train

tensorboard --port=6006 --logdir_spec=.

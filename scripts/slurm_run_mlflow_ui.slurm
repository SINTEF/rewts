#!/bin/bash
#SBATCH --output "mlflow.out"
#SBATCH --job-name MLFlow-TSA
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time 01-00:00:00

cd logs/mlflow || { echo "Failed to change directory."; exit 1; }

mlflow ui

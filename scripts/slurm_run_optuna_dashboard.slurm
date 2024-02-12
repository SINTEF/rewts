#!/bin/bash
#SBATCH --output "optuna-dashboard.out"
#SBATCH --job-name Optuna-Dashboard-TSA
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time 01-00:00:00

cd logs/optuna

optuna-dashboard sqlite:///hyperopt.db

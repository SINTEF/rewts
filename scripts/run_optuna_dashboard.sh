#!/usr/bin/env bash
#SBATCH --output "scripts/slurm_out/optuna-dashboard.out"
#SBATCH --job-name Optuna-Dashboard-TSA
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time 01-00:00:00

PORT=8080
# Check if running under SLURM by testing the presence of the SLURM_JOB_ID variable
if [[ -n "$SLURM_JOB_ID" ]]; then
    # We are under SLURM, use current directory as <root>
    ROOT_DIR=$(pwd)

    # shellcheck disable=SC1091
    source "$ROOT_DIR"/scripts/common.sh

    if [[ -n "${OPTUNA_DASHBOARD_PORT_SLURM}" ]]; then
      PORT=${OPTUNA_DASHBOARD_PORT_SLURM}
    fi
else
    ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )

    # shellcheck disable=SC1091
    source "$ROOT_DIR"/scripts/common.sh

    if [[ -n "${OPTUNA_DASHBOARD_PORT}" ]]; then
      PORT=${OPTUNA_DASHBOARD_PORT}
    fi
fi

cd "$ROOT_DIR" || { echo "Failed to change directory."; exit 1; }

if [[ -z "${OPTUNA_DB}" ]]; then
    OPTUNA_DB=logs/optuna/hyperopt.db
fi

optuna-dashboard sqlite:///"$OPTUNA_DB" --port "${PORT}"

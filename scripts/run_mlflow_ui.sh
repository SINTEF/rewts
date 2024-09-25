#!/usr/bin/env bash
#SBATCH --output "scripts/slurm_out/mlflow.out"
#SBATCH --job-name MLFlow-TSA
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time 01-00:00:00

PORT=5000
# Check if running under SLURM by testing the presence of the SLURM_JOB_ID variable
if [[ -n "$SLURM_JOB_ID" ]]; then
    # We are under SLURM, use current directory as <root>
    ROOT_DIR=$(pwd)

    # shellcheck disable=SC1091
    source "$ROOT_DIR"/scripts/common.sh

    if [[ -n "${MLFLOW_PORT_SLURM}" ]]; then
      PORT=${MLFLOW_PORT_SLURM}
    fi
else
    ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )

    # shellcheck disable=SC1091
    source "$ROOT_DIR"/scripts/common.sh

    if [[ -n "${MLFLOW_PORT}" ]]; then
      PORT=${MLFLOW_PORT}
    fi
fi

cd "$ROOT_DIR"/logs/mlflow || { echo "Failed to change directory."; exit 1; }

if [[ -z "${MLFLOW_STORE_URI}" ]]; then
    mlflow ui --port "${PORT}"
else
    mlflow ui --backend-store-uri "${MLFLOW_STORE_URI}" --port "${PORT}"
fi

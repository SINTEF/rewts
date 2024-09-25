#!/usr/bin/env bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )

# shellcheck disable=SC1091
source "$ROOT_DIR"/scripts/common.sh

# Check if the user provided an argument for user@host
if [ -n "$1" ]; then
  USER_HOST="$1"
else
  USER_HOST="$SSH_USER@$SSH_HOST"
fi

TENSORBOARD_PORT=6006
if [[ -n "${TENSORBOARD_PORT_SLURM}" ]]; then
  TENSORBOARD_PORT=${TENSORBOARD_PORT_SLURM}
fi

MLFLOW_PORT=5000
if [[ -n "${MLFLOW_PORT_SLURM}" ]]; then
  MLFLOW_PORT=${MLFLOW_PORT_SLURM}
fi

OPTUNA_DASHBOARD_PORT=8080
if [[ -n "${OPTUNA_DASHBOARD_PORT_SLURM}" ]]; then
  OPTUNA_DASHBOARD_PORT=${OPTUNA_DASHBOARD_PORT_SLURM}
fi

J_PORT=8898
if [[ -n "${JUPYTER_PORT}" ]]; then
  J_PORT=${JUPYTER_PORT}
fi

# Connect to the server using SSH and forward the required ports
CMD="ssh $USER_HOST -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} -L ${MLFLOW_PORT}:localhost:${MLFLOW_PORT} -L ${OPTUNA_DASHBOARD_PORT}:localhost:${OPTUNA_DASHBOARD_PORT} -L ${J_PORT}:localhost:${J_PORT}"
echo "Running command: ${CMD}"
ssh "$USER_HOST" -L "${TENSORBOARD_PORT}":localhost:"${TENSORBOARD_PORT}" -L "${MLFLOW_PORT}":localhost:"${MLFLOW_PORT}" -L "${OPTUNA_DASHBOARD_PORT}":localhost:"${OPTUNA_DASHBOARD_PORT}" -L "${J_PORT}":localhost:"${J_PORT}"

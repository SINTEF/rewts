#!/bin/bash

# Get the script's directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

# Check if the user provided an argument for user@host
if [ -n "$1" ]; then
  USER_HOST="$1"
else
  USER_HOST="$SSH_USER@$SSH_HOST"
fi

# Connect to the server using SSH and forward the required ports
ssh "$USER_HOST" -L 6007:localhost:6006 -L 5001:localhost:5000 -L 8081:localhost:8080 -L 8889:localhost:8889
# -L = forward ports: tensorboard              mlflow             optuna dashboard       jupyter notebook

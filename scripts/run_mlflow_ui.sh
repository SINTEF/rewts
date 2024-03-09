#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

cd "$SCRIPT_DIR"/../logs/mlflow || { echo "Failed to change directory."; exit 1; }

run_with_conda "mlflow ui"

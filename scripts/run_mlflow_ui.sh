#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/../logs/mlflow

run_with_conda "mlflow ui"

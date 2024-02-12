#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/..

if [[ -z ${OPTUNA_DB+x} ]]; then
  OPTUNA_DB=logs/optuna/hyperopt.db
fi

run_with_conda "optuna-dashboard sqlite:///$OPTUNA_DB"

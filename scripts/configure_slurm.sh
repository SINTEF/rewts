#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/..

cp configs/local/slurm.yaml configs/local/default.yaml
echo "Made SLURM the default launcher: created configs/local/default.yml"
run_with_conda "pip install hydra-submitit-launcher --upgrade"

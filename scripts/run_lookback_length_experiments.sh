#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR/..

source $SCRIPT_DIR/common.sh

chunk_length=1344
chunk_idx_end=32  # will depend on the chunk_length
lookback_length=(288 432 1008 1440)
datamodule="electricity"
models=("xgboost" "tcn" "rnn" "elastic_net")  # datamodule and model are used to name mlflow experiment
model_dirs=("path/to/xgboost/models" "path/to/tcn/models" "path/to/rnn/models" "path/to/elastic_net1/models")

echo "Running lookback experiments with lookback_data_length(s): ${lookback_length[@]}"

# Loop through models
for (( idx=0; idx<${#models[@]}; idx++ )); do
  model=${models[$idx]}
  model_dir=${model_dirs[$idx]}

  # Loop over the lookback lengths
  for i in "${lookback_length[@]}"; do
      echo "Running script for model: $model and lookback: $i"

      python src/eval.py --multirun experiment=chunk_eval_iterative ++datamodule.chunk_length=$chunk_length ++ensemble.lookback_data_length=$i chunk_idx_end=$chunk_idx_end model_type=ensemble ensemble_model_dir=$model_dir +logger=mlflow logger.mlflow.experiment_name=${datamodule}_eval-it_${model}_lookback-$i

      # Check if the script exited successfully
      if [ $? -ne 0 ]; then
          echo "Script failed for model: $model and lookback: $i"
          send_slack_notification "lookback experiment failed for model: $model and lookback: $i"
          exit 1
      fi
  done
done

echo "All iterations completed successfully."

#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"/.. || exit
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

# Default values
default_chunk_lengths="4032,3024,2016,1008,720"
default_models="xgboost,tcn,rnn,elastic_net"
default_datamodule="electricity"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --chunk_lengths) chunk_lengths="$2"; shift ;;
        --models) models="$2"; shift ;;
        --datamodule) datamodule="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use the default values if not provided
chunk_lengths=${chunk_lengths:-$default_chunk_lengths}
models=${models:-$default_models}
datamodule=${datamodule:-$default_datamodule}

IFS=',' read -r -a chunk_lengths_array <<< "$chunk_lengths"
IFS=',' read -r -a models_array <<< "$models"

models=("xgboost" "tcn" "rnn" "elastic_net")

echo "Running chunk_length experiments for datamodule=$datamodule, models: (${models_array[*]}), chunk_lengths: (${chunk_lengths_array[*]})"

for model in "${models_array[@]}"; do
    for chunk_length in "${chunk_lengths_array[@]}"; do
        echo "Running script for: model=$model chunk_length=${chunk_length}"

        {
            scripts/run_iterative_experiment.sh ++datamodule.chunk_length="${chunk_length}" -train datamodule="$datamodule" model="${datamodule}"_"${model}" ~logger -eval +logger=mlflow logger.mlflow.experiment_name="${datamodule}"_eval-it_"${model}"_chunk-length-"${chunk_length}"
        } || {
            echo "Script failed for model=$model chunk_length=${chunk_length}"
            send_slack_notification "chunk_length experiment failed for: model=$model chunk_length=${chunk_length}"
            exit 1
        }
    done
done

echo "All iterations completed successfully."

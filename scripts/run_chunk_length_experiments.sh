#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

source $SCRIPT_DIR/common.sh

# Define start, end, and step for the loop
start=1008  # 6 * 24 * 7
end=4032   # 6 * 24 * 28
step=1008   # 6 * 24 * 7
# Generate a sequence of chunk lengths

chunk_lengths=()  # can add chunk_lengths before the sequence here
# Append the sequence to the array
chunk_lengths+=($(seq $end -$step $start))

chunk_lengths+=(720)

datamodule="electricity"
models=("xgboost" "tcn" "rnn" "elastic_net")

echo "Running chunk_length experiments for datamodule=$datamodule and models=${models[@]} with chunk_lengths: ${chunk_lengths[@]}"

for model in "${models[@]}"; do
    # Loop over the range
    for i in "${chunk_lengths[@]}"; do
        echo "Running script for: model=$model chunk_length=$i"

        scripts/run_iterative_experiment.sh ++datamodule.chunk_length=$i -train datamodule=$datamodule model=${datamodule}_${model} ~logger -eval +logger=mlflow logger.mlflow.experiment_name=${datamodule}_eval-it_${model}_chunk-length-$i

        # Check if the script exited successfully
        if [ $? -ne 0 ]; then
            echo "Script failed for for: model=$model chunk_length=$"
            send_slack_notification "chunk_length experiment failed for: model=$model chunk_length=$"
            exit 1
        fi
    done
done

echo "All iterations completed successfully."

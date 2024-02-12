#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

# usage: all arguments to the script are given to both train.py and eval.py, unless preceded by -train (only for train
# script) or -eval (only for eval script).
# Example: bash scripts/run_iterative_experiment.sh model=arima -train a=b -eval c=d e=f
# Would yield:
#   train args: model=arima a=b
#   eval args:  model=arima c=d e=f

# Initialize variables for holding arguments
train_args=()
eval_args=()
mode=""

# Loop through all the arguments
for arg in "$@"; do
    if [[ $arg == "-train" ]]; then
        mode="train"
    elif [[ $arg == "-eval" ]]; then
        mode="eval"
    else
        # Depending on the mode, assign the argument to the appropriate array
        if [[ $mode == "train" ]]; then
            train_args+=("$arg")
        elif [[ $mode == "eval" ]]; then
            eval_args+=("$arg")
        else
            # If no mode has been set yet, add to both arrays
            train_args+=("$arg")
            eval_args+=("$arg")
        fi
    fi
done

# Convert arrays to strings
train_args_str="${train_args[@]}"
eval_args_str="${eval_args[@]}"

echo "Train args: $train_args_str"
echo "Eval args:  $eval_args_str"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
ensemble_model_dir="logs/train/multiruns/${timestamp}_ensemble"
global_model_dir="logs/train/multiruns/${timestamp}_global"

# Training
echo "Running python src/train.py -m experiment=ensemble hydra.sweep.dir=$ensemble_model_dir $train_args_str"
python src/train.py -m experiment=ensemble hydra.sweep.dir=$ensemble_model_dir $train_args_str
echo "Finished training ensemble models, saved to: $ensemble_model_dir"

# List directories, filter out non-integer names, and find the maximum
max_dir=$(find "$ensemble_model_dir" -maxdepth 1 -type d -exec basename {} \; | grep -E '^[0-9]+$' | sort -n | tail -n 1)

# Check if we found a maximum directory
if [[ -n $max_dir ]]; then
    chunk_idx_end=$max_dir
else
    echo "No run-directories found in $ensemble_model_dir"
    exit 1
fi

echo "Running python src/train.py -m experiment=global_iterative hydra.sweep.dir=$global_model_dir $train_args_str"
python src/train.py -m experiment=global_iterative hydra.sweep.dir=$global_model_dir $train_args_str
echo "Finished training global models, saved to: $global_model_dir"


# Evaluation
echo "Running python src/eval.py --multirun experiment=chunk_eval_iterative chunk_idx_end=$chunk_idx_end model_type=ensemble,global ensemble_model_dir=$ensemble_model_dir global_model_dir=$global_model_dir $eval_args_str"
python src/eval.py --multirun experiment=chunk_eval_iterative chunk_idx_end=$chunk_idx_end model_type=ensemble,global ensemble_model_dir=$ensemble_model_dir global_model_dir=$global_model_dir $eval_args_str
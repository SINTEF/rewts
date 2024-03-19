#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/vars.sh

function run_with_conda() {
    COMMAND="$1"

    # If CONDA_ENV is set
    if [[ -n $CONDA_ENV ]]; then
        # Check if the CONDA_ENV exists
        if conda info --envs | grep -qE "^[[:space:]]*${CONDA_ENV}[[:space:]]+"; then
            echo "Running with conda environment: $CONDA_ENV"
            conda run --live-stream -n "$CONDA_ENV" "$COMMAND"
        else
            echo "Error: CONDA_ENV $CONDA_ENV does not exist."
            exit 1
        fi
    else
        # If CONDA_ENV is not set, run the command without conda
        echo "CONDA_ENV is not set in vars.sh. Running command without conda."
        $COMMAND
    fi
}

function send_slack_notification() {
    local message=$1
    if [[ -z $WEBHOOK_URL ]]; then
        echo "WEBHOOK_URL is not set. No message will be sent to Slack."
        return 1
    fi

    curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"${message}\"}" "$WEBHOOK_URL"
}


function find_highest_chunk() {
    local model_dir="$1"
    local highest_chunk

    # List directories, filter out non-integer names, and find the maximum
    highest_chunk=$(find "$model_dir" -maxdepth 1 -type d -exec basename {} \; | grep -E '^[0-9]+$' | sort -n | tail -n 1)

    if [[ -n $highest_chunk ]]; then
        echo "$highest_chunk"  # Output the result
        return 0  # Success
    else
        echo "No run-directories found in $model_dir" >&2  # Error message to stderr
        return 1  # Signal failure
    fi
}

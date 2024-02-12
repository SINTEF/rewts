#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/..

# This script starts hyperopt jobs in parallel, so that new runs can be started without waiting for all parallel slurm
# jobs to finish first. Note that parallel jobs can also be run by setting the n_jobs argument in the hparams_search
# config.

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
  echo "The script requires two arguments: CONFIG and NJOBS."
  echo "CONFIG should be the name of the YAML configuration file located under config/hparams_search."
  echo "NJOBS specifies the number of jobs to be spawned."
  exit 1
fi

# Required arguments
CONFIG="$1"
NJOBS="$2"

# Optional argument
OVERRIDES=""
if [ "$#" -ge 3 ]; then
  # Shift the first two arguments to capture all remaining arguments as overrides
  shift 2
  OVERRIDES="$@"
fi

# Loop to execute the Python script
for (( i=1; i<=NJOBS; i++ )); do
  # Construct the command
  CMD="nohup python src/train_hopt.py hparams_search=$CONFIG hydra.sweeper.sampler.seed=$i"

  # Add optional overrides if provided
  if [ -n "$OVERRIDES" ]; then
    CMD="$CMD $OVERRIDES"
  fi

  # Execute the command in the background
  #run_with_conda "$CMD &"
  $CMD &

  echo "Spawned process #$i: $CMD"

  # Sleep for a few seconds before the next iteration to allow the first process to create
  sleep 2
done


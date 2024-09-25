#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck disable=SC1091
source "$SCRIPT_DIR"/common.sh

cd "$SCRIPT_DIR"/.. || exit

# This script starts hyperopt jobs in parallel, so that new runs can be started without waiting for all parallel slurm
# jobs to finish first. Note that parallel jobs can also be run by setting the n_jobs argument in the hparams_search
# config, then as a SLURM array where every job must finish before a new parallel array is launched.

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
  echo "The script requires two arguments: CONFIG and NJOBS."
  echo "CONFIG should be the name of the YAML configuration file located under config/hparams_search."
  echo "NJOBS specifies the number of jobs to be spawned."
  echo "Optional arguments:"
  echo "--train_script Which python training script in src/ to use. Defaults to train.py"
  echo "--ntrials Number of total trials to be run, divided among the jobs. If not specified, the config variable "
  echo " hydra.sweeper.n_trials will be used for each job (i.e. total runs = NJOBS * hydra.sweeper.n_trials)"
  exit 1
fi

# Required arguments
CONFIG="$1"
NJOBS="$2"

# Optional arguments
TRAIN_SCRIPT="train.py"
NTRIALS=0

# Shift the first two arguments to process optional and arbitrary arguments
shift 2

# Process the optional arguments and capture the rest as overrides
OVERRIDES=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --train_script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --ntrials)
      NTRIALS="$2"
      shift 2
      ;;
    *)
      OVERRIDES+="$1 "
      shift
      ;;
  esac
done

# Calculate N_TRIAL_JOB if N_TRIALS is defined and not zero
if [ "$NTRIALS" -ne 0 ]; then
  N_TRIALS_JOB=$((NTRIALS / NJOBS))
else
  N_TRIALS_JOB=0
fi

echo "CONFIG: $CONFIG"
echo "NJOBS: $NJOBS"
echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
echo "NTRIALS: $NTRIALS"
echo "OVERRIDES: $OVERRIDES"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# Check if LOGS_ROOT is set and non-empty, use it; otherwise, use the default
base_dir="${LOGS_ROOT:-logs}"
run_dir="${base_dir}/train/multiruns/${timestamp}_hopt-${CONFIG}"

# Loop to execute the Python script
for (( i=1; i<=NJOBS; i++ )); do
  # Construct the command
  CMD="nohup python src/${TRAIN_SCRIPT} hparams_search=${CONFIG} hydra.sweeper.n_jobs=1 hydra.sweeper.sampler.seed=$i hydra.sweep.dir=${run_dir}${i}"

  if [ "$N_TRIALS_JOB" -ne 0 ]; then
    CMD="${CMD} hydra.sweeper.n_trials=${N_TRIALS_JOB}"
  fi

  # Add optional overrides if provided
  if [ -n "$OVERRIDES" ]; then
    CMD="${CMD} ${OVERRIDES}"
  fi

  # Execute the command in the background
  #run_with_conda "$CMD &"
  $CMD &

  echo "Spawned process #$i: $CMD"

  # Sleep for a few seconds before the next iteration to allow the first process to create
  sleep 2
done

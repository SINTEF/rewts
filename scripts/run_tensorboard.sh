#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/common.sh

cd $SCRIPT_DIR/..

LOGS_BASE_DIR="logs/train/runs"
TENSORBOARD_LOGDIR=""

# Function to check if a date is within the given range
is_date_in_range() {
  local date="$1"
  local start_date="$2"
  local end_date="$3"

  if [[ -z "${start_date}" && -z "${end_date}" ]]; then
    return 0 # No range specified, include all dates
  elif [[ -z "${start_date}" && "${date}" < "${end_date}" ]]; then
    return 0 # Only end date specified, include dates before end date
  elif [[ "${start_date}" < "${date}" && -z "${end_date}" ]]; then
    return 0 # Only start date specified, include dates after start date
  elif [[ "${start_date}" < "${date}" && "${date}" < "${end_date}" ]]; then
    return 0 # Both start and end dates specified, include dates within the range
  else
    return 1 # Date is outside the specified range
  fi
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --base)
      LOGS_BASE_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    --time)
      TIME_RANGE="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done


# Parse time range argument and filter directories if provided
if [[ -n "$TIME_RANGE" ]]; then
  if [[ "$TIME_RANGE" =~ ^([<>])([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]]; then
    sign="${BASH_REMATCH[1]}"
    date_value="${BASH_REMATCH[2]}"
    if [[ "${sign}" == ">" ]]; then
      start_date="${date_value}"
    elif [[ "${sign}" == "<" ]]; then
      end_date="${date_value}"
    fi
  elif [[ "$TIME_RANGE" =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]]; then
    start_date="${BASH_REMATCH[1]}"
    end_date="${BASH_REMATCH[2]}"
  else
    echo "Invalid time range format. Please use 'YYYY-mm-dd_YYYY-mm-dd', '>YYYY-mm-dd', or '<YYYY-mm-dd'."
    exit 1
  fi

  # Iterate through the directories and filter by the time range
  for dir in "${LOGS_BASE_DIR}"/*; do
    if [[ -d "${dir}" ]]; then
      folder_date="$(basename "${dir}" | cut -d'_' -f1)"
      if is_date_in_range "${folder_date}" "${start_date}" "${end_date}"; then
        TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR},${dir}"
      fi
    fi
  done

  # Remove the leading comma from the logdir string
  TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR#,}"

  if [[ -z "${TENSORBOARD_LOGDIR}" ]]; then
    echo "No logs were found that fit the provided time filter. TensorBoard was not started."
    exit 1
  fi
else
  TENSORBOARD_LOGDIR="${LOGS_BASE_DIR}"
fi

# Start TensorBoard
run_with_conda "tensorboard --logdir ${TENSORBOARD_LOGDIR} --port 6006"
#TB_PID=$!
#echo "TensorBoard started with PID ${TB_PID} on port 6006 with logdir ${TENSORBOARD_LOGDIR}"
#echo "Use 'kill ${TB_PID}' to stop TensorBoard"

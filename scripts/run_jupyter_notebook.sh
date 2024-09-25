#!/usr/bin/env bash

PORT=8888
ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )

# shellcheck disable=SC1091
source "$ROOT_DIR"/scripts/common.sh

if [[ -n "${JUPYTER_PORT}" ]]; then
  PORT=${JUPYTER_PORT}
fi

cd "$ROOT_DIR" || { echo "Failed to change directory."; exit 1; }

jupyter notebook --no-browser --port "${PORT}"

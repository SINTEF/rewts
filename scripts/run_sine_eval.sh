#!/bin/bash
# Run from root folder with: bash scripts/run_sine_eval.sh
# Run all sine experiment evaluations

# Global model

## individual chunks
python src/eval.py -m experiment=sine_eval model_dir=$1 +datamodule=sine_data_train,sine_data_test datamodule.chunk_idx='range(8)'
## all chunks concatenated
python src/eval.py -m experiment=sine_eval model_dir=$1 +datamodule=sine_data_train,sine_data_test eval.plot.title_add_metrics=False

# Ensemble model

## individual chunks
python src/eval.py -m experiment=sine_eval model_dir=$2 +datamodule=sine_data_train,sine_data_test datamodule.chunk_idx='range(8)' ++ensemble.fit_weights_every=10000
## all chunks concatenated
python src/eval.py -m experiment=sine_eval model_dir=$2 +datamodule=sine_data_train,sine_data_test ++ensemble.fit_weights_every=1 eval.plot.title_add_metrics=False

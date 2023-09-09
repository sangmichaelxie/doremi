#!/bin/bash

# This is a sample of the constants file. Please write any env variables here
# and rename the file to constants.sh

CACHE=/path/to/cache
DOREMI_DIR=/path/to/this/repo
PILE_DIR=/path/to/pile
PREPROCESSED_PILE_DIR=/path/to/preprocessed  # will be created by scripts/run_filter_domains.sh
MODEL_OUTPUT_DIR=/path/to/model_output_dir
WANDB_API_KEY=key  # Weights and Biases key for logging
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
source ${DOREMI_DIR}/venv/bin/activate  # if you installed doremi in venv
export WANDB_API_KEY=${WANDB_API_KEY}

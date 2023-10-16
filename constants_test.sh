#!/bin/bash

# This is a sample of the constants file. Please write any env variables here
# and rename the file constants.sh

CACHE=/home/jzhanggr/doremi/cache_test
DOREMI_DIR=/home/jzhanggr/doremi
PILE_DIR=/home/jzhanggr/doremi/data/pile_v1
PREPROCESSED_PILE_DIR=/home/jzhanggr/doremi/preprocessed_test  # will be created by scripts/run_filter_domains.sh
MODEL_OUTPUT_DIR=/home/jzhanggr/doremi/model_output_dir_test
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
#conda activate doremi  # if you installed doremi in venv
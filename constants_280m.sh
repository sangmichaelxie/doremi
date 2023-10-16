#!/bin/bash

# This is a sample of the constants file. Please write any env variables here
# and rename the file constants.sh

ROOT=/export/project/shizhe/doremi_run/doremi
CACHE=$ROOT/cache_test_nfs
#CACHE=/home/sdiaoaa/cache_test_nfs
DOREMI_DIR=$ROOT
PILE_DIR=/export/project/shizhe/pile/pile_v1
PREPROCESSED_PILE_DIR=/export/project/shizhe/doremi_run/doremi/preprocessed_sample_280m/  # will be created by scripts/run_filter_domains.sh
MODEL_OUTPUT_DIR=/export/project/shizhe/doremi_run/doremi/model_output_dir_280m_20k
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
#conda activate doremi  # if you installed doremi in venv
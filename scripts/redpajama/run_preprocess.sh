#!/bin/bash
DOMAIN=$1
SUBSET=$2
NUM_SUBSETS=$3

set -x

# load global parameters
source constants.sh

mkdir -p ${CACHE}/logs

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE

cd ${DOREMI_DIR}

python scripts/redpajama/preprocess.py \
	--dataset_dir ${RP_DIR} \
	--output_dir ${CACHE}/preprocessed_rp \
        --cache_dir ${CACHE} \
	--domain $DOMAIN \
        --max_length 2048 \
        --nproc 95 \
        --num_subsets ${NUM_SUBSETS} \
        --subset ${SUBSET} \
	--num_validation_examples 1000000 \
        --seed 111

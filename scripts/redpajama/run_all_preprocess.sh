#!/bin/bash

# load global parameters
source constants.sh

set -x

# this can be parallelized for faster execution
NUM_SUBSETS=10
for DOMAIN in 'c4' 'arxiv' 'github' 'wikipedia' 'book' 'stackexchange' 'common_crawl';
do
for ((SUBSET=0; SUBSET<${NUM_SUBSETS}; SUBSET++));
do
bash scripts/redpajama/run_preprocess.sh ${DOMAIN} ${SUBSET} ${NUM_SUBSETS}
done
done

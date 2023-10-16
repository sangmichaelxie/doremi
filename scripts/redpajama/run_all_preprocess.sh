#!/bin/bash

# load global parameters
source constants.sh

set -x

NUM_SUBSETS=10
DOMAIN=c4
for ((SUBSET=0; SUBSET<${NUM_SUBSETS}; SUBSET++));
do
bash scripts/redpajama/run_preprocess.sh ${DOMAIN} ${SUBSET} ${NUM_SUBSETS}
done

NUM_SUBSETS=5
for DOMAIN in 'arxiv' 'github' 'wikipedia' 'book' 'stackexchange';
do
for ((SUBSET=0; SUBSET<${NUM_SUBSETS}; SUBSET++));
do
bash scripts/redpajama/run_preprocess.sh ${DOMAIN} ${SUBSET} ${NUM_SUBSETS}
done
done


NUM_SUBSETS=10
for DOMAIN in 'common_crawl';
do
for ((SUBSET=0; SUBSET<${NUM_SUBSETS}; SUBSET++));
do
bash scripts/redpajama/run_preprocess.sh ${DOMAIN} ${SUBSET} ${NUM_SUBSETS}
done
done


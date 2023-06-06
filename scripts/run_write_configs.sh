#!/bin/bash

source constants.sh

python scripts/write_config.py --config_name pile_baseline_50kvocab --preprocessed_dir ${PREPROCESSED_PILE_DIR} --cache_dir ${CACHE}
# python scripts/write_config.py --config_name pile_uniform
# python scripts/write_config.py --config_name doremi_280M_256kvocab


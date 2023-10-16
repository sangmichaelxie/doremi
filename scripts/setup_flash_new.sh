#!/bin/bash

source constants.sh
# export PYTHONNOUSERSITE=True
# python -m pip install -e .
# git clone git@github.com:/HazyResearch/flash-attention 
# git clone flash-attention @ git+ https://github.com/Dao-AILab/flash-attention.git@d2f4324f4c56e017fbf22dc421943793a8ca6c3b

FLASHATTENTION=flash-attention-2.2.2
cd ${DOREMI_DIR}/$FLASHATTENTION && python setup.py install
cd ${DOREMI_DIR}/$FLASHATTENTION/csrc/fused_dense_lib && pip install .
cd ${DOREMI_DIR}/$FLASHATTENTION/csrc/xentropy && pip install .
cd ${DOREMI_DIR}/$FLASHATTENTION/csrc/rotary && pip install .
cd ${DOREMI_DIR}/$FLASHATTENTION/csrc/layer_norm && python -m pip install . --verbose
    


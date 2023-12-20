#!/bin/bash

source constants.sh

pip install -e .
pip uninstall -y ninja && pip install ninja
cd ${DOREMI_DIR}/flash-attention && python setup.py install
cd ${DOREMI_DIR}/flash-attention/csrc/fused_dense_lib && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/xentropy && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/rotary && pip install .
cd ${DOREMI_DIR}/flash-attention/csrc/layer_norm && pip install .



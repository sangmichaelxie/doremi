#!/bin/bash

#
# Sample baseline model run of a 120M model with the same number of non-embedding parameters as the 280M model in the DoReMi paper. Not the same as DoReMi paper since the paper uses 256k vocab size.
#


# load global parameters
source constants.sh
pip install -e .

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb

PREPROCESSED_DATA=${PREPROCESSED_RP_DIR}
PREPROCESSED_CACHE=${CACHE}/preprocessed_rp

if [ ! -d "${PREPROCESSED_CACHE}" ]; then
    mkdir -p ${CACHE}/preprocessed_cache
    cp -r ${PREPROCESSED_DATA} ${PREPROCESSED_CACHE}
fi

NAME=rp_uniform_280M
accelerate launch \
    --config_file accelerate_config.yml \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 60200 \
    doremi/train.py \
    --dataset_name redpajama \
    --model_type gpt_neox_flash \
    --tokenizer_name togethercomputer/RedPajama-INCITE-Base-7B-v0.1 \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_CACHE} \
    --domain_config_path configs/rp_uniform.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 2048 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --max_steps 200000 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 10000 \
    --learning_rate 1e-3 \
    --lr_end 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-8 \
    --lr_scheduler_name linear_warmup_cosine \
    --warmup_ratio 0.06 \
    --run_name ${NAME} \
    --seed 1111 \
    --logging_strategy steps \
    --logging_steps 100 \
    --logging_first_step \
    --report_to wandb \
    --optim adafactor \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --bf16 \
    --config_overrides="n_embd=1024,n_layer=16,n_head=16"


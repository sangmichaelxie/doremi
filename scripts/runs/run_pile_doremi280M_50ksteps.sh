#!/bin/bash

#
# Sample run of DoReMi 280M model with same number of total params as 280M model in DoReMi paper
#

# load global parameters
source constants.sh

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb


PREPROCESSED_DATA=${PREPROCESSED_PILE_DIR}
PREPROCESSED_CACHE=${CACHE}/preprocessed_cache/perdomain_pile_preprocessed

if [ ! -d "${PREPROCESSED_CACHE}" ]; then
    mkdir -p ${CACHE}/preprocessed_cache
    cp -r ${PREPROCESSED_DATA} ${PREPROCESSED_CACHE}
fi

NAME=pile_doremi_280M_50k
accelerate launch \
    --config_file accelerate_config.yml \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --main_process_port 60600 \
    doremi/train.py \
    --dataset_name pile \
    --model_type gpt_neox \
    --tokenizer_name gpt2 \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_CACHE} \
    --domain_config_path configs/pile_uniform.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 1024 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 2 \
    --max_steps 50000 \
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
    --doremi_optimizer doremiv1 \
    --reweight_eta 1 \
    --reweight_eps 1e-4 \
    --train_domain_weights_tmp_file ${CACHE}/tmp_${NAME}_domain_weight \
    --reweight_domains \
    --remove_unused_columns=False \
    --reference_model_name_or_path ${MODEL_OUTPUT_DIR}/pile_baseline_280M_50k/checkpoint-50000 \
    --bf16 \
    --config_overrides="max_position_embeddings=1024,hidden_size=1024,num_hidden_layers=18,num_attention_heads=16,intermediate_size=4096,vocab_size=50257"

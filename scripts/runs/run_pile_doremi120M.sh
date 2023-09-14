#!/bin/bash

#
# Sample run of DoReMi with a 120M proxy model.
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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:12288


ROUND=${1-1}
# name of the initial reference weights
REFERENCE_WEIGHTS_NAME=${2:-"pile_baseline_50kvocab_nopack"}
# name of the reference model to load (can be different from reference weights in later rounds)
REFERENCE_MODEL_NAME=${3:-"pile_baseline_50kvocab_nopack"}
OTHER_ARGS=$4

NAME=pile_doremi_r${ROUND}_120M_ref:${REFERENCE_WEIGHTS_NAME}_120M
accelerate launch \
    --config_file accelerate_config.yml \
    --num_processes 8 \
    --multi_gpu \
    --num_machines 1 \
    --main_process_port 60600 \
    doremi/train.py \
    --dataset_name pile \
    --model_type gpt_flash \
    --tokenizer_name togethercomputer/RedPajama-INCITE-Base-7B-v0.1 \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_PILE_DIR} \
    --domain_config_path configs/pile_uniform.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 1024 \
    --per_device_train_batch_size 64 \
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
    --lr_scheduler_name linear_warmup_exponential \
    --warmup_ratio 0.06 \
    --run_name ${NAME} \
    --seed 1111 \
    --logging_strategy steps \
    --logging_steps 100 \
    --logging_first_step \
    --report_to wandb \
    --optim adamw_torch_fused \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --doremi_optimizer doremiv1 \
    --reweight_eta 1.0 \
    --reweight_eps 1e-3 \
    --reweight_domains \
    --remove_unused_columns=False \
    --reference_model_name_or_path ${MODEL_OUTPUT_DIR}/${REFERENCE_MODEL_NAME}/checkpoint-200000 \
    --bf16 \
    --shuffle \
    --config_overrides="n_positions=1024,n_embd=768,n_layer=12,n_head=12,rotary_emb_fraction=0.25,tie_word_embeddings=True,scale_attn_by_inverse_layer_idx=False,embd_pdrop=0.0,resid_pdrop=0.0,attn_pdrop=0.0,eos_token_id=0,bos_token_id=0,max_position_embeddings=0,vocab_size=50277" \
    ${OTHER_ARGS}


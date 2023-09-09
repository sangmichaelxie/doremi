#!/bin/bash

#
# Sample baseline model run of a 120M model on The Pile, with the same number of non-embedding parameters as the 280M model in the DoReMi paper. Not the same as DoReMi paper since the paper uses 256k vocab size. Append "eval" to the command to do perplexity and few-shot evaluation.
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

DOMAIN_CONFIG_NAME=${1:-pile_baseline_50kvocab_nopack}
arg=${2:-""} # set to eval to run eval

if [[ "${arg}" == "eval" ]]; then
   ADDITIONAL_ARGS="--evaluation_strategy steps --per_device_eval_batch_size 32 --do_train false --remove_unused_columns=False --downstream_datasets trivia_qa,web_questions,lambada,natural_questions,squad_v2 --eval_all_checkpoints"
else
    ADDITIONAL_ARGS=""
fi

NAME=${DOMAIN_CONFIG_NAME}_120M
accelerate launch \
    --config_file accelerate_config.yml \
    --num_machines 1 \
    --num_processes 8 \
    --multi_gpu \
    --main_process_port 60100 \
    doremi/train.py \
    --dataset_name pile \
    --model_type gpt_flash \
    --tokenizer_name togethercomputer/RedPajama-INCITE-Base-7B-v0.1 \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_DATA} \
    --domain_config_path configs/${DOMAIN_CONFIG_NAME}.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 1024 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --max_steps 200000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --per_device_eval_batch_size 32 \
    --remove_unused_columns=False \
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
    --bf16 \
    --shuffle \
    --config_overrides="n_positions=1024,n_embd=768,n_layer=12,n_head=12,rotary_emb_fraction=0.25,tie_word_embeddings=True,scale_attn_by_inverse_layer_idx=False,embd_pdrop=0.0,resid_pdrop=0.0,attn_pdrop=0.0,eos_token_id=0,bos_token_id=0,max_position_embeddings=0,vocab_size=50277" \
    ${ADDITIONAL_ARGS}


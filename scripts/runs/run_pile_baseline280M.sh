#!/bin/bash
export WANDB_MODE=disabled
#
# Sample baseline model run of a 280M model with the same number of non-embedding parameters as the 280M model in the DoReMi paper. Not the same as DoReMi paper since the paper uses 256k vocab size.
#


# load global parameters
source constants.sh
# pip install -e .

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb
export NCCL_IBEXT_DISABLE=1
export PDSH_RCMD_TYPE=ssh
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=br0 
export NCCL_DEBUG=INFO

PREPROCESSED_DATA=${PREPROCESSED_PILE_DIR}
# PREPROCESSED_CACHE=${DOREMI_DIR}/preprocessed/
PREPROCESSED_CACHE=${DOREMI_DIR}/preprocessed/

if [ ! -d "${PREPROCESSED_CACHE}" ]; then
    mkdir -p ${CACHE}/preprocessed_cache
    cp -r ${PREPROCESSED_DATA} ${PREPROCESSED_CACHE}
fi

arg=${1:-""} # set to eval to run eval

if [[ "${arg}" == "eval" ]]; then
    ADDITIONAL_ARGS="--evaluation_strategy steps --per_device_eval_batch_size 32 --do_train false --remove_unused_columns=False"
else
    ADDITIONAL_ARGS=""
fi

log_dir=output_dir/test_doremi_train

if [ -d "${log_dir}" ]; then
	echo "${log_dir} exists"
else
	mkdir -p ${log_dir}
fi

NAME=pile_baseline_280M_256kvocab
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
# --ddp_timeout \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file accelerate_config.yml \
    --multi_gpu \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 60200 \
    doremi/train.py \
    --dataset_name pile \
    --model_type gpt_flash \
    --tokenizer_name gpt2 \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_CACHE} \
    --domain_config_path configs/pile_baseline_256kvocab.json \
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
    --config_overrides="n_positions=1024,n_embd=1024,n_layer=18,n_head=16" \
    ${ADDITIONAL_ARGS} | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err 

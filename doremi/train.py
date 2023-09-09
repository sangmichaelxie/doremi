#!/usr/bin/env python
# coding=utf-8

# Training code adapted from the HuggingFace Team run_clm.py code.

# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
from pathlib import Path
import os
import sys
import json
import numpy as np

import datasets
import torch

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import TrainerState
from transformers.trainer import TRAINER_STATE_NAME
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from doremi.training_args import ModelArguments, DataTrainingArguments, FullTrainingArguments
import doremi.dataloader as data_utils
from doremi.trainer import DoReMiTrainer
import doremi.models as doremi_models
try:
    from flash_attn.models.gpt_neox import gpt_neox_config_to_gpt2_config
except Exception:
    pass


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FullTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    num_skip_examples = 0
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            state = TrainerState.load_from_json(str(Path(last_checkpoint) / TRAINER_STATE_NAME))
            global_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
            num_skip_examples = state.global_step * global_batch_size
            logger.info(f"Skipping {num_skip_examples} examples")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        if model_args.model_type == 'gpt_neox_flash':
            config = gpt_neox_config_to_gpt2_config(config)
            config.use_flash_attn = True
            config.fused_mlp = True
            config.fused_bias_fc = True
            config.fused_dropout_add_ln = True
            config.pad_vocab_size_multiple = 8
            config.activation_function = 'gelu_new'
            config.n_inner = None
            # disable absolute
            config.max_position_embeddings = 0
    else:
        if model_args.model_type == 'gpt_flash':
            config = GPT2Config(
                    vocab_size=50257, n_positions=2048, n_embd=2048,
                    n_layer=24, n_head=16,
                    scale_attn_by_inverse_layer_idx=True,
                    rotary_emb_fraction=0.5,
                    use_flash_attn=True, fused_mlp=True,
                    fused_bias_fc=True, fused_dropout_add_ln=True,
                    pad_vocab_size_multiple=8)
            # disable absolute
            config.max_position_embeddings = 0
        elif model_args.model_type == 'gpt_neox_flash':
            # convert to GPT2 config
            config = CONFIG_MAPPING['gpt_neox']()
            config = gpt_neox_config_to_gpt2_config(config)
            config.use_flash_attn = True
            config.fused_mlp = True
            config.fused_bias_fc = True
            config.fused_dropout_add_ln = True
            config.pad_vocab_size_multiple = 8
            config.activation_function = 'gelu_new'
            config.n_inner = None
            # disable absolute
            config.max_position_embeddings = 0
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.model_max_length = data_args.max_token_length

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if model_args.model_type in {'gpt_flash', 'gpt_neox_flash'}:
            model = doremi_models.GPTFlashAttnLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
    else:
        if model_args.model_type in {'gpt_flash', 'gpt_neox_flash'}:
            model = doremi_models.GPTFlashAttnLMHeadModel(config)
        else:
            model = AutoModelForCausalLM.from_config(config)

        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    with open(training_args.domain_config_path, 'r') as f:
        domain_config = json.load(f)

    train_domain_weights_dict = domain_config['train_domain_weights']
    eval_domain_weights_dict = domain_config['eval_domain_weights']
    # whenever we convert dict to array, we sort by key
    domain_list = list(sorted(train_domain_weights_dict.keys()))

    if training_args.reweight_domains:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if model_args.model_type in {'gpt_flash', 'gpt_neox_flash'}:
            model_cls = doremi_models.GPTFlashAttnLMHeadModel
            reference_model = model_cls.from_pretrained(
                training_args.reference_model_name_or_path,
                config=config)
        else:
            model_cls = AutoModelForCausalLM
            reference_model = model_cls.from_pretrained(
                training_args.reference_model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
            )
        for param in reference_model.parameters():
            param.requires_grad = False
        model.reference_model = reference_model
        total_domain_weight = sum(train_domain_weights_dict.values())
        model.register_buffer('train_domain_weights', torch.tensor(
                [train_domain_weights_dict[domain] / total_domain_weight for domain in domain_list]))
        model.register_buffer('avg_domain_weights', model.train_domain_weights.clone())
        model.register_buffer('perdomain_scores', torch.ones(len(train_domain_weights_dict)) * np.log(len(tokenizer)))
        model.register_buffer('update_counter', torch.tensor(1))

    else:
        reference_model = None

    if training_args.do_train:
        # data script could change tokenizer shape
        train_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.dataset_dir,
                domain_weights_dict=train_domain_weights_dict,
                dataset_name=data_args.dataset_name,
                cache_dir=model_args.cache_dir,
                split='train',
                max_samples=data_args.max_train_samples,
                add_domain_id=data_args.add_domain_id,
                domain_weight_buffer_handle=None,
                seed=training_args.seed,
                tokenizer=tokenizer,
                shuffle=data_args.shuffle,
                num_skip_examples=num_skip_examples,
                shard_reversal=training_args.reweight_domains,
                keep_in_memory=data_args.keep_in_memory)

    if training_args.do_eval:
        if data_args.eval_dataset_dir is None:
            data_args.eval_dataset_dir = data_args.dataset_dir
        if data_args.eval_dataset_name is None:
            data_args.eval_dataset_name = data_args.dataset_name

        eval_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.eval_dataset_dir,
                domain_weights_dict=eval_domain_weights_dict,
                dataset_name=data_args.eval_dataset_name,
                cache_dir=model_args.cache_dir,
                split='validation',
                add_domain_id=data_args.add_domain_id,
                max_samples=data_args.max_eval_samples,
                tokenizer=tokenizer,
                no_interleave=True,
                keep_in_memory=data_args.keep_in_memory)

    # turn off find unused parameters
    training_args.ddp_find_unused_parameters = False

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings.weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    torch.cuda.empty_cache()

    # Initialize our Trainer
    trainer = DoReMiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_utils.get_data_collator(tokenizer, do_padding=data_args.do_padding, max_length=data_args.max_token_length),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if training_args.reweight_domains:
            avg_domain_weights_dict = {}
            for i in range(len(model.avg_domain_weights)):
                domain_name = domain_list[i]
                metrics[f'avg_domain_weight:{domain_name}'] = model.avg_domain_weights[i].item()
                avg_domain_weights_dict[domain_name] = model.avg_domain_weights[i].item()

            # save avg domain weights to json
            avg_domain_weights_file = Path(training_args.output_dir) / 'avg_domain_weights.json'
            with open(avg_domain_weights_file, 'w') as f:
                json.dump(avg_domain_weights_dict, f, indent=2)

            # also save to configs dir
            config_dict = {"train_domain_weights": avg_domain_weights_dict,
                           "eval_domain_weights": avg_domain_weights_dict}
            config_dict_file = Path(__file__).parent.parent / 'configs' / f"{Path(training_args.output_dir).name}.json"
            with open(config_dict_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if training_args.eval_all_checkpoints:
            checkpoint_dir_list = trainer.get_all_checkpoints(training_args.output_dir)
        else:
            checkpoint_dir_list = [get_last_checkpoint(training_args.output_dir)]

        for checkpoint_dir in checkpoint_dir_list:
            trainer.load_checkpoint(checkpoint_dir)
            state = TrainerState.load_from_json(str(Path(checkpoint_dir) / TRAINER_STATE_NAME))
            trainer.state.global_step = state.global_step

            if not training_args.skip_perplexity_eval:
                metrics = trainer.evaluate()
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            if training_args.downstream_datasets is not None:
                dataset_names = training_args.downstream_datasets.split(',')
                downstream_metrics = trainer.evaluate_fewshot(
                        dataset_names,
                        max_samples=data_args.max_downstream_samples,
                        num_shots=training_args.downstream_num_shots)
                trainer.log_metrics("eval", downstream_metrics)
                trainer.save_metrics("eval", downstream_metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

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
import math
from pathlib import Path
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import json
import numpy as np
import pickle

import datasets
import evaluate
import torch
from datasets import load_dataset
import numpy as np

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import doremi.dataloader as data_utils
from doremi.trainer import DoReMiTrainer
from doremi.models import GPT2LMHeadModelFast, GPTNeoXForCausalLMFast


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: str = field(
        default='.', metadata={"help": "Path to the dataset directory."}
    )
    dataset_name: str = field(
        default='pile', metadata={"help": "Name of the dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_token_length: int = field(
        default=1024,
        metadata={
            "help": (
                "Input sequence length after tokenization. "
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_padding: bool = field(
        default=False, metadata={"help": "Pad the inputs."}
    )
    add_domain_id: bool = field(
        default=False, metadata={"help": "Add domain id to examples (when it's not already in the data)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class FullTrainingArguments(TrainingArguments):
    domain_config_path: str = field(
        default='.', metadata={"help": "Path to the domain config file."}
            )
    lr_end: float = field(
            default=1e-3,
            metadata={"help": "The final learning rate of the learning rate scheduler."},
    )
    reweight_domains: bool = field(
        default=False, metadata={"help": "Do reweighting."}
    )
    reweight_eta: float = field(
            default=1.0,
            metadata={"help": "Learning rate for reweighting."},
    )
    reweight_eps: float = field(
            default=1e-4,
            metadata={"help": "Smoothing parameter for reweighting."},
    )
    doremi_optimizer: str = field(
        default='doremiv1', metadata={"help": "Optimizer for DoReMi."}
    )
    reference_model_name_or_path: str = field(
        default='.', metadata={"help": "Path to the reference model."}
    )
    lr_scheduler_name: str = field(
        default=None, metadata={"help": "Custom LR scheduler name (linear_warmup_exponential, linear_warmup_cosine)"}
    )
    train_domain_weights_tmp_file: str = field(
        default=None, metadata={"help": "Path to the temporary file for training domain weights."}
    )


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
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
        if model_args.model_type == 'gpt2':
            model = GPT2LMHeadModelFast(config)
        elif model_args.model_type == 'gpt_neox':
            model = GPTNeoXForCausalLMFast(config)
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
    num_domains = len(domain_list)

    if training_args.reweight_domains:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if model_args.model_type == 'gpt2':
            model_cls = GPT2LMHeadModelFast
        elif model_args.model_type == 'gpt_neox':
            model_cls = GPTNeoXForCausalLMFast
        else:
            model_cls = AutoModelForCausalLM
            
        reference_model = model_cls.from_pretrained(
            training_args.reference_model_name_or_path,
            from_tf=bool(".ckpt" in training_args.reference_model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
        )
        for param in reference_model.parameters():
            param.requires_grad = False
        reference_model.eval()
        model.reference_model = reference_model
        model.register_buffer('train_domain_weights', torch.tensor(
                [train_domain_weights_dict[domain] for domain in domain_list]))
        model.register_buffer('avg_domain_weights', model.train_domain_weights.clone())
        model.register_buffer('perdomain_scores', torch.ones(len(train_domain_weights_dict)) * np.log(len(tokenizer)))
        model.register_buffer('update_counter', torch.tensor(1))

    else:
        reference_model = None

    # turn off find unused parameters
    training_args.ddp_find_unused_parameters = False

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings.weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))


    if training_args.do_train:
        train_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.dataset_dir,
                domain_weights_dict=train_domain_weights_dict,
                dataset_name=data_args.dataset_name,
                cache_dir=model_args.cache_dir,
                split='train',
                sharded=True,
                max_samples=data_args.max_train_samples,
                add_domain_id=data_args.add_domain_id,
                tmp_file=None)

    if training_args.do_eval:
        eval_dataset = data_utils.get_preprocessed_mixed_dataset(
                preprocessed_dir=data_args.dataset_dir,
                domain_weights_dict=eval_domain_weights_dict,
                dataset_name=data_args.dataset_name,
                cache_dir=model_args.cache_dir,
                split='validation',
                sharded=False,
                add_domain_id=data_args.add_domain_id,
                max_samples=data_args.max_eval_samples)

        def preprocess_logits_for_metrics(logits, labels):
            return logits

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_prediction):
            preds = eval_prediction.predictions.logits.argmax(-1)
            domain_ids = eval_prediction.predictions.domain_ids
            labels = eval_prediction.label_ids
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            metrics = {}
            for domain_idx in data_args.num_domains:
                domain_mask = (domain_ids == domain_idx)
                domain_labels = labels[domain_mask]
                domain_preds = preds[domain_mask]
                domain_metrics = metric.compute(predictions=domain_preds, references=domain_labels)
                for k, v in domain_metrics.items():
                    metrics[f'{domain_list[domain_idx]}_{k}'] = v

            full_metrics = metric.compute(predictions=preds, references=labels)
            metrics = {**metrics, **full_metrics}
            return metrics

    torch.cuda.empty_cache()

    # Initialize our Trainer
    trainer = DoReMiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_utils.get_data_collator(tokenizer, do_padding=data_args.do_padding),
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
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
            for i in range(len(model.avg_train_domain_weights)):
                domain_name = domain_list[i]
                metrics[f'avg_domain_weight:{domain_name}'] = model.avg_domain_weights[i].item()
                avg_domain_weights_dict[domain_name] = model.avg_domain_weights[i].item()

            # save avg domain weights to json
            avg_domain_weights_file = Path(training_args.output_dir) / 'avg_domain_weights.json'
            with open(avg_domain_weights_file, 'w') as f:
                json.dump(avg_domain_weights_dict, f, indent=2)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

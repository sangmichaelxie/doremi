import math
import warnings
import json
import re
from pathlib import Path
import wandb
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from datasets import IterableDataset
from transformers import Trainer
from transformers.utils import ExplicitEnum, is_torch_tpu_available
from transformers.optimization import get_scheduler
from transformers.utils import logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import (
        has_length,
        denumpify_detensorize,
        EvalLoopOutput,
        enable_full_determinism,
        set_seed,
        get_last_checkpoint,
        PREFIX_CHECKPOINT_DIR
)
from transformers.trainer_pt_utils import find_batch_size

from doremi.eval_datasets import get_eval_dataset


logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


class LinearWarmupExponentialLR(LRScheduler):
    """
    Exponential LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # figure out decay rate to use to get within 1e-10 of lr_end at end of training
            gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                      for base_lr in self.base_lrs]
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for base_lr, gamma in zip(self.base_lrs, gammas)]


class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]


class ExtendedSchedulerType(ExplicitEnum):
    LINEAR_WARMUP_EXPONENTIAL = "linear_warmup_exponential"
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"


# extend scheduler function mapping
TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
        ExtendedSchedulerType.LINEAR_WARMUP_EXPONENTIAL: LinearWarmupExponentialLR,
        ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR
}


def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps=0,
    num_training_steps=0,
    lr_end=1e-4,
):

    try:
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]
    except ValueError:
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_end=lr_end)


class DoReMiTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.args.domain_config_path, 'r') as f:
            self.domain_config = json.load(f)

        self.train_domain_weights_dict = self.domain_config['train_domain_weights']
        self.eval_domain_weights_dict = self.domain_config['eval_domain_weights']

        self.eval_domain_list = list(sorted(self.eval_domain_weights_dict.keys()))
        self.domain_list = list(sorted(self.train_domain_weights_dict.keys()))
        self.sampling_weights = torch.tensor([self.train_domain_weights_dict[domain] for domain in self.domain_list])

        self.pertoken_scores = []
        self.token_masks = []
        self.domain_ids = []

        # we will take care of skipping in dataloader
        self.args.ignore_data_skip = True

    def write_weights(self, weights):
        self.model.update_counter += 1
        self.model.train_domain_weights[:] = weights.float()
        self.model.avg_domain_weights[:] = (self.model.avg_domain_weights * (self.model.update_counter - 1) + weights) / self.model.update_counter

    def read_weights(self):
        return self.model.train_domain_weights.clone()

    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def create_optimizer(self):
        self.optimizer = super().create_optimizer()

        optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(self.args)
        if optimizer_cls.__name__ == "Adafactor":
            self.optimizer.beta1 = self.args.adam_beta1
            for param_group in self.optimizer.param_groups:
                param_group['beta1'] = self.args.adam_beta1
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_name is not None:
                lr_scheduler_name = self.args.lr_scheduler_name
            else:
                lr_scheduler_name = self.args.lr_scheduler_type
            self.lr_scheduler = get_scheduler_extended(
                lr_scheduler_name,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                lr_end=self.args.lr_end,
            )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, return_pertoken_losses=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        inputs['return_pertoken_losses'] = return_pertoken_losses

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    def update_domain_weights(self, scores, scores_mask, domain_ids):
        wandb_log_dict = {}
        train_domain_weights = self.read_weights()

        scores = scores.detach()
        domain_ids = domain_ids.detach()

        if self.args.doremi_optimizer == 'doremiv1':
            perdomain_scores = []
            for domain_id in range(len(train_domain_weights)):
                domain_mask = (domain_ids == domain_id)
                perdomain_scores_mask = scores_mask[domain_mask]
                if domain_mask.sum() > 0:
                    curr_domain_scores = torch.clip(scores[domain_mask][perdomain_scores_mask], min=0).mean()
                else:
                    curr_domain_scores = self.model.perdomain_scores[domain_id]
                perdomain_scores.append(curr_domain_scores)
            self.model.perdomain_scores[:] = torch.tensor(perdomain_scores).float()
            log_new_train_domain_weights = torch.log(train_domain_weights) + self.args.reweight_eta * self.model.perdomain_scores
            log_new_train_domain_weights = log_new_train_domain_weights - torch.logsumexp(log_new_train_domain_weights, dim=0)
            train_domain_weights = (1-self.args.reweight_eps) * torch.exp(log_new_train_domain_weights) + self.args.reweight_eps / len(log_new_train_domain_weights)
            self.write_weights(train_domain_weights)
        else:
            raise ValueError(f"DoReMi optimizer {self.args.doremi_optimizer} not supported")

        for domain_idx in range(len(train_domain_weights)):
            domain_name = self.domain_list[domain_idx]
            wandb_log_dict[f'avg_domain_weights/{domain_name}'] = self.model.avg_domain_weights[domain_idx].item()
            wandb_log_dict[f'train_domain_weights/{domain_name}'] = self.model.train_domain_weights[domain_idx].item()
            wandb_log_dict[f'perdomain_scores/{domain_name}'] = self.model.perdomain_scores[domain_idx].item()
        wandb_log_dict['max_domain_id'] = domain_ids.max().item()
        wandb.log(wandb_log_dict, commit=False)

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.args.reweight_domains:
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True, return_pertoken_losses=True)
                pertoken_loss = outputs.pertoken_loss
                reference_pertoken_loss = outputs.reference_pertoken_loss
                token_mask = outputs.token_mask
                excess_loss = pertoken_loss - reference_pertoken_loss

            if self.is_local_process_zero():
                gathered_excess_losses = [
                        torch.zeros_like(excess_loss) for _ in range(self.args.world_size)
                        ]
                dist.gather(excess_loss, gathered_excess_losses, dst=0)
                gathered_excess_losses = torch.cat(gathered_excess_losses, dim=0)

                gathered_token_mask = [
                        torch.zeros_like(token_mask) for _ in range(self.args.world_size)
                        ]
                dist.gather(token_mask, gathered_token_mask, dst=0)
                gathered_token_mask = torch.cat(gathered_token_mask, dim=0)

                gathered_domain_id = [
                        torch.zeros_like(inputs['domain_ids']) for _ in range(self.args.world_size)
                        ]
                dist.gather(inputs['domain_ids'], gathered_domain_id, dst=0)
                gathered_domain_id = torch.cat(gathered_domain_id, dim=0)

                self.pertoken_scores.append(gathered_excess_losses.detach())
                self.token_masks.append(gathered_token_mask.detach())
                self.domain_ids.append(gathered_domain_id.detach())

                if len(self.pertoken_scores) == self.args.gradient_accumulation_steps:
                    pertoken_scores = torch.cat(self.pertoken_scores, dim=0)
                    token_masks = torch.cat(self.token_masks, dim=0).bool()
                    domain_ids = torch.cat(self.domain_ids, dim=0)

                    # update domain weights
                    self.update_domain_weights(pertoken_scores, token_masks, domain_ids)

                    self.pertoken_scores = []
                    self.token_masks = []
                    self.domain_ids = []
            else:
                dist.gather(excess_loss, dst=0)
                dist.gather(token_mask, dst=0)
                dist.gather(inputs['domain_ids'], dst=0)

            if self.args.doremi_optimizer == 'doremiv1':
                # compute the rescaled loss, divide by domain weights
                train_domain_weights = self.read_weights().to(pertoken_loss.device).float()

                # if doing non-uniform sampling, normalize by inverse sampling weight
                train_domain_weights = train_domain_weights / self.sampling_weights.to(train_domain_weights.device)
                train_domain_weights = train_domain_weights / train_domain_weights.sum()
                curr_domain_weights = train_domain_weights[inputs['domain_ids']].unsqueeze(-1).expand_as(pertoken_loss).detach()

                curr_domain_weights = curr_domain_weights * token_mask

                # renormalize
                normalizer = curr_domain_weights.detach().sum()
                # gather normalizer across GPUs
                dist.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
                # scale by world size because DDP averages gradients
                normalizer = torch.clip(normalizer, min=1e-10) / self.args.world_size

                token_mask = token_mask.detach().type(pertoken_loss.dtype)
                loss = (pertoken_loss * curr_domain_weights.detach()).sum() / normalizer
            else:
                raise ValueError(f"doremi_optimizer {self.args.doremi_optimizer} is not supported")
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def load_checkpoint(self, resume_from_checkpoint=None):
        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(None)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is None:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)

        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None and not is_sagemaker_mp_enabled() and self.args.deepspeed is None:
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, self.args.device)
            self.model_wrapped = self.model

    def get_all_checkpoints(self, folder):
        _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
        folder = Path(folder)
        checkpoints = [
            path
            for path in folder.iterdir()
            if _re_checkpoint.search(path.name) is not None and path.is_dir()
        ]
        checkpoints = list(sorted(checkpoints, key=lambda x: int(x.name.split('-')[1])))
        checkpoints = [str(path) for path in checkpoints]
        return checkpoints

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Computes per-domain log-perplexity, uniformly averaged log-perplexity, and worst-case log-perplexity
        """
        args = self.args

        if prediction_loss_only:
            # hack - don't do prediction loss only
            prediction_loss_only = None

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        loss_fn = nn.CrossEntropyLoss(reduction='sum')

        losses = torch.zeros(len(self.eval_domain_list)).cuda()
        tokencounts = torch.zeros(len(self.eval_domain_list)).cuda()
        examplecounts = torch.zeros(len(self.eval_domain_list)).cuda()
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader)):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            domain_ids = inputs["domain_ids"].to(loss.device)

            if is_torch_tpu_available():
                xm.mark_step()

            if isinstance(logits, tuple):
                logits = logits[0]

            # compute losses per domain
            for domain_idx, domain_name in enumerate(self.eval_domain_list):
                domain_mask = (domain_ids == domain_idx)
                examplecounts[domain_idx] = examplecounts[domain_idx] + domain_mask.sum()

                if domain_mask.sum() > 0:
                    domain_labels = labels[domain_mask]
                    domain_preds = logits[domain_mask]
                    domain_labels = domain_labels[:, 1:].contiguous().view(-1)
                    domain_preds = domain_preds[:, :-1, :].contiguous().view(-1, domain_preds.size(-1))
                    losses[domain_idx] = losses[domain_idx] + loss_fn(domain_preds, domain_labels)
                    tokencounts[domain_idx] = tokencounts[domain_idx] + (domain_labels != -100).sum()

        torch.distributed.all_reduce(losses)
        torch.distributed.all_reduce(tokencounts)
        torch.distributed.all_reduce(examplecounts)

        # losses/preds/labels on CPU (final containers)
        per_domain_losses = {domain_name: losses[domain_idx].item()
                             for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}
        per_domain_tokencounts = {domain_name: tokencounts[domain_idx].item()
                                  for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}
        per_domain_examplecounts = {domain_name: examplecounts[domain_idx].item()
                                    for domain_idx, domain_name in enumerate(self.eval_domain_list) if tokencounts[domain_idx] > 0}

        # normalize
        per_domain_losses = {domain_name: per_domain_losses[domain_name] / per_domain_tokencounts[domain_name]
                             for domain_name in per_domain_losses.keys()}

        metrics = {f"{domain_name}:log_perplexity": per_domain_losses[domain_name]
                   for domain_name in per_domain_losses.keys()}
        metrics["uniform_avg_log_perplexity"] = np.mean(list(per_domain_losses.values()))
        metrics["worst_case_log_perplexity"] = np.amax(list(per_domain_losses.values()))

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=sum(list(per_domain_examplecounts.values())))

    def evaluate_fewshot(self, dataset_names, ignore_keys=None, metric_key_prefix="eval", num_shots=1, max_samples=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        max_token_length = self.tokenizer.model_max_length
        # prepare tokenizer
        tokenizer = self.tokenizer
        tokenizer_padding_side = tokenizer.padding_side
        tokenizer_truncation_side = tokenizer.truncation_side
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'

        all_metrics = {}

        for dataset_name in dataset_names:
            logger.info(f"Evaluating {dataset_name}...")

            # we pass in num_shots because some datasets are only 0 shot
            data_dict = get_eval_dataset(dataset_name, num_shots=num_shots, seed=self.args.seed)

            dataset_train = data_dict['dataset_train']
            dataset_val = data_dict['dataset_val']
            top_k = data_dict['top_k']
            top_p = data_dict['top_p']
            temperature = data_dict['temperature']
            prompt_transform = data_dict['prompt_transform']
            eval_func = data_dict['eval_func']
            pred_postprocess_func = data_dict['pred_postprocess_func']
            num_shots = data_dict['num_shots']
            max_new_tokens = data_dict['max_new_tokens']
            shuffle_train = data_dict['shuffle_train']

            # use training set as few-shot examples, shuffle first
            if dataset_train is not None and shuffle_train:
                dataset_train = dataset_train.shuffle(seed=self.args.seed)

            # select first num examples
            if max_samples is not None:
                dataset_val = dataset_val.select(range(max_samples))

            # shard the dataset
            if dataset_train is not None:
                dataset_train = dataset_train.shard(num_shards=self.args.world_size, index=self.args.process_index)
            dataset_val = dataset_val.shard(num_shards=self.args.world_size, index=self.args.process_index)

            def few_shot_generator(ds=None):
                while True:
                    curr_exs = []
                    if num_shots == 0:
                        yield curr_exs
                        continue

                    for ex in ds:
                        curr_exs.append(ex)

                        if len(curr_exs) == num_shots:
                            yield curr_exs
                            curr_exs = []

            fewshot_train_dataset = IterableDataset.from_generator(
                    few_shot_generator, gen_kwargs={'ds': dataset_train})

            def prompt_generator(fewshot_train_ds, val_ds):
                for ex, context_exs in zip(val_ds, fewshot_train_ds):
                    ex_dict = prompt_transform(ex, context_exs)
                    yield ex_dict

            def data_collator(batch):
                # self.tokenizer is the HF tokenizer
                # tokenizer is either HF tokenizer or SPM tokenizer
                collated_batch = {k: [f[k] for f in batch] for k in batch[0].keys()}
                # will do left truncation
                tokenized = tokenizer(collated_batch['prompt'], padding=False, truncation=True)

                collated_batch['input_ids'] = torch.tensor(tokenized['input_ids'])[:, -(max_token_length-max_new_tokens):]
                collated_batch['attention_mask'] = torch.tensor(tokenized['attention_mask'])[:, -(max_token_length-max_new_tokens):]
                return collated_batch

            fewshot_val_dataset = IterableDataset.from_generator(
                    prompt_generator, gen_kwargs={'fewshot_train_ds': fewshot_train_dataset, 'val_ds': dataset_val})

            dataloader = DataLoader(
                    fewshot_val_dataset,
                    batch_size=1,  # batch size 1 avoids left padding
                    collate_fn=data_collator,
                    num_workers=1,
                    pin_memory=self.args.dataloader_pin_memory)

            # prepare model
            args = self.args
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)

            # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
            # while ``train`` is running, cast it to the right dtype first and then put on device
            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)
            model.eval()

            # TODO put this somewhere else?
            model.config.pad_token_id = model.config.eos_token_id

            num_correct = torch.tensor(0.0).cuda()
            num_examples = torch.tensor(0.0).cuda()
            # fewshot eval loop
            for step, inputs in tqdm(enumerate(dataloader)):
                num_examples += len(inputs['input_ids'])
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        gen_tokens = model.generate(
                                input_ids=inputs['input_ids'].cuda(),
                                max_length=inputs['input_ids'].shape[1]+max_new_tokens,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature)

                gen_text = tokenizer.batch_decode(gen_tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                for prompt, pred, answer in zip(inputs['prompt'], gen_text, inputs['answer']):
                    pred = pred_postprocess_func(pred)
                    if eval_func(answer, pred, prompt,
                                 model=model,
                                 tokenizer=tokenizer,
                                 inputs=inputs,
                                 trainer=self):
                        num_correct += 1
                        print(f"\033[0;32m CORRECT \033[0m: {prompt}\033[0;32m{pred}\033[0m |  Answer: {answer}\n")
                    else:
                        print(f"\033[91m INCORRECT \033[0m: {prompt}\033[91m{pred}\033[0m |  Answer: {answer}\n")

            torch.distributed.all_reduce(num_correct)
            torch.distributed.all_reduce(num_examples)
            accuracy = 100 * (num_correct / num_examples)

            metrics = {'accuracy': accuracy, 'num_correct': num_correct, 'num_examples': num_examples}

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_{num_shots}-shot:{dataset_name}"):
                    metrics[f"{metric_key_prefix}_{num_shots}-shot:{dataset_name}:{key}"] = metrics.pop(key)

            all_metrics.update(metrics)

        # comput average metrics across datasets
        avg_metrics = defaultdict(list)
        for key in all_metrics:
            if key.endswith('accuracy'):
                avg_metrics['accuracy'].append(all_metrics[key])
            if key.endswith('num_correct'):
                avg_metrics['num_correct'].append(all_metrics[key])
            if key.endswith('num_examples'):
                avg_metrics['num_examples'].append(all_metrics[key])

        avg_metrics = {key: np.mean(val_list) for key, val_list in avg_metrics.items()}

        for key in avg_metrics.keys():
            all_metrics[f"{metric_key_prefix}_{num_shots}-shot:avg:{key}"] = avg_metrics[key]

        # gather and compute metrics
        output = EvalLoopOutput(predictions=None, label_ids=None, metrics=all_metrics, num_samples=None)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # restore tokenizer settings
        tokenizer.padding_side = tokenizer_padding_side
        tokenizer.truncation_side = tokenizer_truncation_side

        return output.metrics

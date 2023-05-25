import math
import warnings
import json
import wandb
import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
from transformers import Trainer
from transformers.utils import ExplicitEnum
from transformers.optimization import get_scheduler
from torch import nn
from transformers.utils import logging
from transformers.trainer import is_sagemaker_mp_enabled

logger = logging.get_logger(__name__)



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
        # figure out decay rate to use to get within 1e-10 of lr_end at end of training
        self.gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_training_steps - self.num_warmup_steps))
                       for base_lr in self.base_lrs]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for gamma, base_lr in zip(self.base_lrs, self.gammas)]


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

        if self.last_epoch == 0 or self.last_epoch > self.num_training_steps:
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

        self.domain_list = list(sorted(self.train_domain_weights_dict.keys()))
        train_domain_weights = [self.train_domain_weights_dict[domain] for domain in self.domain_list]

        self.pertoken_scores = []
        self.token_masks = []
        self.domain_ids = []
        self.update_counter = 1
        self.accum_counter = 0
        self.avg_train_domain_weights = torch.tensor(train_domain_weights)

        if self.args.reweight_domains:
            if self.args.doremi_optimizer == 'doremiv1':
                # initial domain weights
                self.train_domain_weights = torch.tensor(train_domain_weights)
                self.perdomain_scores = torch.ones(len(self.train_domain_weights_dict)) * np.log(len(self.tokenizer))
            else:
                raise ValueError(f"DoReMi optimizer {self.args.doremi_optimizer} not supported")

    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
           setattr(self, k, v)

    def create_optimizer(self):
        self.optimizer = super().create_optimizer()

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
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

        if return_pertoken_losses:
            inputs['return_pertoken_losses'] = True

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
        elif return_pertoken_losses:
            return (loss,
                    outputs['pertoken_loss'],
                    outputs['reference_pertoken_loss'],
                    outputs['token_mask'])
        else:
            return loss

    def update_domain_weights(self, scores, scores_mask, domain_ids):
        scores = scores.detach()
        domain_ids = domain_ids.detach()

        if self.args.doremi_optimizer == 'doremiv1':
            perdomain_scores = []
            for domain_id in range(len(self.train_domain_weights)):
                domain_mask = (domain_ids == domain_id)
                perdomain_scores_mask = scores_mask[domain_mask]
                # print type of perdomain_scores_mask
                if domain_mask.sum() > 0:
                    curr_domain_scores = torch.clip(scores[domain_mask][perdomain_scores_mask], min=0).mean()
                else:
                    curr_domain_scores = self.perdomain_scores[domain_id]
                perdomain_scores.append(curr_domain_scores)
            self.perdomain_scores = torch.tensor(perdomain_scores)
            log_new_train_domain_weights = torch.log(self.train_domain_weights) + self.args.reweight_eta * self.perdomain_scores
            new_train_domain_weights = nn.functional.softmax(log_new_train_domain_weights, dim=0)
            # make sure it sums to 1
            new_train_domain_weights = new_train_domain_weights / new_train_domain_weights.sum()
            new_train_domain_weights = (1-self.args.reweight_eps) * new_train_domain_weights + self.args.reweight_eps / len(new_train_domain_weights)

            self.train_domain_weights = new_train_domain_weights
        else:
            raise ValueError(f"DoReMi optimizer {self.args.doremi_optimizer} not supported")

        self.update_counter += 1
        self.avg_train_domain_weights = (self.avg_train_domain_weights * (self.update_counter - 1) + new_train_domain_weights) / self.update_counter

        wandb_log_dict = {}
        for domain_idx in range(len(new_train_domain_weights)):
            domain_name = self.domain_list[domain_idx]
            wandb_log_dict[f'avg_domain_weights/{domain_name}'] = self.avg_train_domain_weights[domain_idx].item()
            wandb_log_dict[f'train_domain_weights/{domain_name}'] = new_train_domain_weights[domain_idx].item()
            wandb_log_dict[f'perdomain_scores/{domain_name}'] = self.perdomain_scores[domain_idx].item()
        wandb_log_dict[f'max_domain_id'] = domain_ids.max().item()
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
                loss, pertoken_loss, reference_pertoken_loss, token_mask = self.compute_loss(model, inputs, return_pertoken_losses=True)
                excess_loss = pertoken_loss - reference_pertoken_loss

            if self.is_local_process_zero():
                with torch.no_grad():
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
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.reweight_domains and self.args.doremi_optimizer == 'doremiv1':
            # compute the rescaled loss, divide by domain weights
            train_domain_weights_gpu = self.train_domain_weights.to(pertoken_loss.device)
            curr_domain_weights = train_domain_weights_gpu[inputs['domain_ids']].unsqueeze(-1).expand_as(pertoken_loss).detach()
            token_mask = token_mask.detach().type(pertoken_loss.dtype)
            loss = (pertoken_loss * curr_domain_weights * token_mask).sum() / (curr_domain_weights * token_mask).sum()

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

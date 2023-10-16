from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
try:
    from flash_attn.models.gpt import GPTLMHeadModel as GPTLMHeadModelFlash
    from flash_attn.models.gpt import shard_state_dict_tp
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    from flash_attn.utils.pretrained import state_dict_from_pretrained
    from flash_attn.models.opt import remap_state_dict_hf_opt
    from flash_attn.models.gptj import remap_state_dict_hf_gptj
    from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
except Exception:
    from torch.nn import CrossEntropyLoss


import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalLMOutputWithDomainIDs(CausalLMOutputWithCrossAttentions):
    domain_ids: Optional[torch.LongTensor] = None
    reference_pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    token_mask: Optional[torch.BoolTensor] = None  # 1 for tokens that are not padding


class GPTFlashAttnLMHeadModel(GPTLMHeadModelFlash):

    def __init__(self, config, process_group=None, device=None, dtype=None):
        super().__init__(config, process_group=process_group, device=device, dtype=dtype)
        self.ignore_index = -100
        self.loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        self.pertoken_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        self.reference_model = None
        self._keys_to_ignore_on_save = []

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        domain_ids: Optional[torch.LongTensor] = None,
        return_pertoken_losses: Optional[bool] = False,
        inference_params: Optional[dict] = None,
        last_token_only: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithDomainIDs]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not return_pertoken_losses:
            lm_logits = super().forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    inference_params=inference_params,
                    last_token_only=last_token_only).logits

            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # Flatten the tokens
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = None

            if not return_dict:
                output = (lm_logits, None, None, None, domain_ids, None, None, None) 
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithDomainIDs(
                loss=loss,
                logits=lm_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                domain_ids=domain_ids)
        else:
            lm_logits = super().forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    inference_params=inference_params,
                    last_token_only=last_token_only).logits

            loss = None
            pertoken_loss = None
            reference_pertoken_loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # Flatten the tokens
                ignore_index = -100
                pertoken_loss = self.pertoken_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))
                token_mask = shift_labels.ne(ignore_index).float()

                loss = pertoken_loss.sum() / token_mask.sum()

                # run reference model forward to get pertoken_loss
                if self.reference_model is not None:
                    self.reference_model.eval()
                    reference_outputs = self.reference_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        head_mask=head_mask,
                        past_key_values=past_key_values,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        domain_ids=domain_ids,
                        return_pertoken_losses=True,
                        position_ids=position_ids,
                        inference_params=inference_params,
                        last_token_only=last_token_only,
                    )
                    reference_pertoken_loss = reference_outputs['pertoken_loss']

            if not return_dict:
                output = (lm_logits, None, None, None, domain_ids, pertoken_loss, reference_pertoken_loss, token_mask) 
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithDomainIDs(
                loss=loss,
                logits=lm_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                domain_ids=domain_ids,
                pertoken_loss=pertoken_loss,
                reference_pertoken_loss=reference_pertoken_loss,
                token_mask=token_mask)

    @classmethod
    def from_pretrained(cls, model_name, config, *args, strict=True, device=None, dtype=None,
                        world_size=1, rank=0, **kwargs):
        """
        Instantiate a GPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)
        # Load state_dict in cpu because we already initialized the model in GPU, and we don't
        # want extra stuff taking up more GPU memory
        state_dict = state_dict_from_pretrained(model_name, device='cpu', dtype=dtype)
        if model_name.startswith('gpt2'):
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        elif model_name.startswith('facebook/opt'):
            state_dict = remap_state_dict_hf_opt(state_dict, config)
        elif model_name.startswith('EleutherAI/gpt-j-'):
            state_dict = remap_state_dict_hf_gptj(state_dict, config)
            strict = False  # We have rotary_emb.inf_freq buffers not in the GPT-J checkpoint
        elif model_name.startswith('EleutherAI/gpt-neox-') or model_name.startswith('EleutherAI/pythia-'):
            state_dict = remap_state_dict_hf_gpt_neox(state_dict, config)
        else:
            pass

        if world_size > 1:
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model

from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Attention, GPT2Block
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import bitsandbytes as bnb
import xformers.ops as xops
from xformers.ops import LowerTriangularMask

import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalLMOutputWithDomainIDs(CausalLMOutputWithCrossAttentions):
    domain_ids: Optional[torch.LongTensor] = None
    reference_pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    token_mask: Optional[torch.BoolTensor] = None  # 1 for tokens that are not padding


class GPT2AttentionFast(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.dropout_prob = config.attn_pdrop

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        out = xops.memory_efficient_attention(
                query=query,
                key=key,
                value=value,
                attn_bias=xops.LowerTriangularMask(),
                p=self.dropout_prob)
        return out, None


class GPT2BlockFast(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = GPT2AttentionFast(config, layer_idx=layer_idx)


class GPT2PreTrainedModelFast(GPT2PreTrainedModel):
    _no_split_modules = ["GPT2BlockFast"]

class GPT2ModelFast(GPT2Model):

    def __init__(self, config):
        GPT2PreTrainedModelFast.__init__(self, config)
        self.embed_dim = config.hidden_size

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.wte = bnb.nn.StableEmbedding(config.vocab_size, self.embed_dim)
        self.wpe = bnb.nn.StableEmbedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2BlockFast(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

class GPT2LMHeadModelFast(GPT2LMHeadModel):

    def __init__(self, config):
        GPT2PreTrainedModelFast.__init__(self, config)
        self.transformer = GPT2ModelFast(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.reference_model = None

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        domain_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_pertoken_losses: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithDomainIDs]:
        if not return_pertoken_losses:
            out = super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return CausalLMOutputWithDomainIDs(
                loss=out.loss,
                logits=out.logits,
                past_key_values=out.past_key_values,
                hidden_states=out.hidden_states,
                attentions=out.attentions,
                cross_attentions=out.cross_attentions,
                domain_ids=domain_ids,
            )
        else:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_head.weight.device)

            lm_logits = self.lm_head(hidden_states)

            loss = None
            pertoken_loss = None
            reference_pertoken_loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                ignore_index = -100
                loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
                pertoken_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))
                token_mask = shift_labels.ne(ignore_index).float()

                loss = (pertoken_loss * token_mask).sum() / token_mask.sum()
                pertoken_loss = pertoken_loss * token_mask

                # run reference model forward to get pertoken_loss
                if self.reference_model is not None:
                    self.reference_model.eval()
                    reference_outputs = self.reference_model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        labels=labels,
                        domain_ids=domain_ids,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        return_pertoken_losses=True,
                    )
                    reference_pertoken_loss = reference_outputs['pertoken_loss']

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithDomainIDs(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
                domain_ids=domain_ids,
                pertoken_loss=pertoken_loss,
                reference_pertoken_loss=reference_pertoken_loss,
                token_mask=token_mask)

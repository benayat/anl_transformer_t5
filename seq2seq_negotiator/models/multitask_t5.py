
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import ModelOutput

ACTION_TO_ID = {"ACCEPT": 0, "OFFER": 1}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}

@dataclass
class NegotiationMultiTaskOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    action_loss: Optional[torch.Tensor] = None
    offer_loss: Optional[torch.Tensor] = None
    action_logits: Optional[torch.Tensor] = None

class MultiTaskT5ForNegotiation(T5ForConditionalGeneration):
    config_class = T5Config

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.classifier_dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.1)))
        self.action_classifier = nn.Linear(config.d_model, 2)
        self.reset_action_head()
        if not hasattr(self.config, "action_loss_weight"):
            self.config.action_loss_weight = 1.0
        if not hasattr(self.config, "offer_loss_weight"):
            self.config.offer_loss_weight = 1.0

    def reset_action_head(self) -> None:
        std = float(getattr(self.config, "initializer_factor", 1.0)) * (float(self.config.d_model) ** -0.5)
        nn.init.normal_(self.action_classifier.weight, mean=0.0, std=std)
        nn.init.zeros_(self.action_classifier.bias)

    def action_head_is_finite(self) -> bool:
        return bool(torch.isfinite(self.action_classifier.weight).all().item() and torch.isfinite(self.action_classifier.bias).all().item())

    def maybe_repair_action_head(self) -> bool:
        if self.action_head_is_finite():
            return False
        self.reset_action_head()
        return True

    def _pool_encoder_hidden(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def assert_all_parameters_finite(self) -> None:
        bad = []
        for name, param in self.named_parameters():
            if not torch.isfinite(param.detach()).all():
                bad.append(name)
                if len(bad) >= 5:
                    break
        if bad:
            raise RuntimeError(f"Non-finite parameters detected: {bad}")

    def _is_generation_mode(self, *, input_ids, action_labels, offer_labels, kwargs) -> bool:
        if action_labels is not None or offer_labels is not None:
            return False
        if input_ids is None:
            return True
        generation_keys = {"encoder_outputs", "decoder_input_ids", "decoder_attention_mask", "past_key_values", "use_cache", "cache_position"}
        return any(k in kwargs and kwargs[k] is not None for k in generation_keys)

    @staticmethod
    def _clean_super_kwargs(kwargs: dict) -> dict:
        cleaned = dict(kwargs)
        cleaned.pop("return_dict", None)
        return cleaned

    @torch.no_grad()
    def debug_forward_probe(self, *, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
        self.eval()
        self.assert_all_parameters_finite()
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool_encoder_hidden(enc.last_hidden_state, attention_mask)
        logits = self.action_classifier(self.classifier_dropout(pooled))
        return {
            "action_head_finite": self.action_head_is_finite(),
            "encoder_finite": bool(torch.isfinite(enc.last_hidden_state).all().item()),
            "pooled_finite": bool(torch.isfinite(pooled).all().item()),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "logits_shape": tuple(logits.shape),
        }

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                action_labels: Optional[torch.Tensor] = None, offer_labels: Optional[torch.Tensor] = None, **kwargs):
        if self._is_generation_mode(input_ids=input_ids, action_labels=action_labels, offer_labels=offer_labels, kwargs=kwargs):
            super_kwargs = self._clean_super_kwargs(kwargs)
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **super_kwargs)

        if input_ids is None:
            raise ValueError("input_ids must not be None in multitask mode")

        self.assert_all_parameters_finite()
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool_encoder_hidden(encoder_outputs.last_hidden_state, attention_mask)
        action_logits = self.action_classifier(self.classifier_dropout(pooled))

        action_loss = None
        if action_labels is not None:
            if action_labels.dtype != torch.long:
                action_labels = action_labels.long()
            action_loss = nn.functional.cross_entropy(action_logits, action_labels)

        offer_loss = None
        if offer_labels is not None:
            valid_offer_examples = (offer_labels != -100).any(dim=1)
            if valid_offer_examples.any():
                super_kwargs = self._clean_super_kwargs(kwargs)
                outputs = super().forward(
                    input_ids=input_ids[valid_offer_examples],
                    attention_mask=attention_mask[valid_offer_examples] if attention_mask is not None else None,
                    labels=offer_labels[valid_offer_examples],
                    return_dict=True,
                    **super_kwargs,
                )
                offer_loss = outputs.loss

        loss = None
        action_w = float(getattr(self.config, "action_loss_weight", 1.0))
        offer_w = float(getattr(self.config, "offer_loss_weight", 1.0))
        if action_loss is not None and offer_loss is not None:
            loss = action_w * action_loss + offer_w * offer_loss
        elif action_loss is not None:
            loss = action_w * action_loss
        elif offer_loss is not None:
            loss = offer_w * offer_loss

        return NegotiationMultiTaskOutput(loss=loss, action_loss=action_loss, offer_loss=offer_loss, action_logits=action_logits)

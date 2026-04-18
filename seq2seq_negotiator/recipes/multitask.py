from __future__ import annotations

from dataclasses import dataclass
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, T5Config

from seq2seq_negotiator.core.dataset_views import load_named_view
from seq2seq_negotiator.models.multitask_t5 import ID_TO_ACTION, MultiTaskT5ForNegotiation
from .base import TrainRecipeConfig


@dataclass
class NegotiationDataCollator:
    tokenizer: any
    label_pad_token_id: int = -100

    def __call__(self, features):
        base_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(base_features, padding=True, return_tensors="pt")
        max_offer_len = max(len(f["offer_labels"]) for f in features)
        padded = []
        for f in features:
            labels = list(f["offer_labels"])
            padded.append(labels + [self.label_pad_token_id] * (max_offer_len - len(labels)))
        batch["offer_labels"] = torch.tensor(padded, dtype=torch.long)
        batch["action_labels"] = torch.tensor([int(f["action_labels"]) for f in features], dtype=torch.long)
        return batch


@dataclass
class MultiTaskPredictor:
    model: any
    tokenizer: any
    device: str

    def predict_batch(self, source_texts, *, max_source_length: int, max_new_tokens: int, num_beams: int):
        enc = self.tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            action_ids = outputs.action_logits.argmax(dim=-1).tolist()
            actions = [ID_TO_ACTION[int(i)] for i in action_ids]
            offer_bodies = [""] * len(actions)
            offer_indices = [i for i, a in enumerate(actions) if a == "OFFER"]
            if offer_indices:
                sub_input_ids = enc["input_ids"][offer_indices]
                sub_attention_mask = enc["attention_mask"][offer_indices]
                out = self.model.generate(input_ids=sub_input_ids, attention_mask=sub_attention_mask, max_new_tokens=max_new_tokens, num_beams=num_beams)
                texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                for idx, text in zip(offer_indices, texts):
                    t = (text or "").strip()
                    if t.upper().startswith("ACTION OFFER"):
                        t = t[len("ACTION OFFER") :].strip()
                    elif t.upper().startswith("O "):
                        t = t[2:].strip()
                    elif t.upper() == "O":
                        t = ""
                    offer_bodies[idx] = t
        return actions, offer_bodies


class MultiTaskRecipe:
    name = "multitask"

    def load_training_views(self, *, dataset_dir, stage: str, serialization_version: str = "v1") -> DatasetDict:
        suffix = "_v2" if serialization_version == "v2" else ""
        view = ("multitask_teacher" if stage == "warmstart" else "multitask_main") + suffix
        return load_named_view(dataset_dir=dataset_dir, view_name=view, splits=("train", "valid"))

    def make_train_config(self, *, init_model: str, stage: str, max_source_length: int, max_target_length: int, action_loss_weight: float, offer_loss_weight: float, reset_action_head_on_load: bool, serialization_version: str = "v1") -> TrainRecipeConfig:
        return TrainRecipeConfig(init_model=init_model, stage=stage, max_source_length=max_source_length, max_target_length=max_target_length, action_loss_weight=action_loss_weight, offer_loss_weight=offer_loss_weight, reset_action_head_on_load=reset_action_head_on_load, serialization_version=serialization_version)

    def build_model_and_tokenizer(self, *, model_source: str, cfg: TrainRecipeConfig):
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        tokenizer.truncation_side = "left"
        config = T5Config.from_pretrained(model_source)
        config.action_loss_weight = cfg.action_loss_weight
        config.offer_loss_weight = cfg.offer_loss_weight
        model = MultiTaskT5ForNegotiation.from_pretrained(model_source, config=config)
        if cfg.reset_action_head_on_load:
            model.reset_action_head()
        if not model.action_head_is_finite():
            model.maybe_repair_action_head()
        return model, tokenizer

    def build_collator(self, *, tokenizer, model):
        return NegotiationDataCollator(tokenizer=tokenizer)

    def tokenize_views(self, *, raw_ds: DatasetDict, tokenizer, cfg: TrainRecipeConfig) -> DatasetDict:
        def _normalize(values):
            out = []
            for v in values:
                if v is None:
                    out.append("")
                else:
                    s = str(v)
                    out.append("" if s.lower() == "nan" else s)
            return out

        action_col = "action_label"
        offer_col = "offer_target_text"
        cols = set(raw_ds["train"].column_names)
        if action_col not in cols:
            action_candidates = [c for c in ("teacher_action_label", "main_action_label") if c in cols]
            offer_candidates = [c for c in ("teacher_offer_target_text", "main_offer_target_text") if c in cols]
            if not action_candidates or not offer_candidates:
                raise KeyError("Could not resolve multitask target columns")
            action_col = action_candidates[0]
            offer_col = offer_candidates[0]

        def _tok(batch):
            model_inputs = tokenizer(batch["source_text"], max_length=cfg.max_source_length, truncation=True)
            offer_targets = _normalize(batch[offer_col])
            offer_enc = tokenizer(text_target=offer_targets, max_length=cfg.max_target_length, truncation=True)
            offer_labels, action_labels = [], []
            for action, ids in zip(batch[action_col], offer_enc["input_ids"]):
                action = str(action).upper()
                action_labels.append(0 if action == "ACCEPT" else 1)
                if action == "OFFER":
                    offer_labels.append(ids)
                else:
                    offer_labels.append([-100] * len(ids))
            model_inputs["action_labels"] = action_labels
            model_inputs["offer_labels"] = offer_labels
            return model_inputs

        return raw_ds.map(_tok, batched=True, remove_columns=raw_ds["train"].column_names, load_from_cache_file=True, keep_in_memory=False)

    def build_predictor(self, *, model_dir, device: str = "auto"):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = MultiTaskT5ForNegotiation.from_pretrained(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return MultiTaskPredictor(model=model, tokenizer=tokenizer, device=device)

    def build_predictor_from_model(self, *, model, tokenizer):
        device = str(next(model.parameters()).device)
        return MultiTaskPredictor(model=model, tokenizer=tokenizer, device=device)

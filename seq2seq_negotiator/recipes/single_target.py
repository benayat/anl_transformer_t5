from __future__ import annotations

from dataclasses import dataclass
import torch
from datasets import DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from seq2seq_negotiator.core.dataset_views import load_named_view
from seq2seq_negotiator.core.metrics import parse_single_target_action_text
from .base import TrainRecipeConfig


@dataclass
class SingleTargetPredictor:
    model: any
    tokenizer: any
    device: str

    def predict_batch(self, source_texts, *, max_source_length: int, max_new_tokens: int, num_beams: int):
        enc = self.tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams)
            texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        pred_actions, pred_offer_bodies = [], []
        for text in texts:
            action, offer_body = parse_single_target_action_text(text)
            pred_actions.append(action)
            pred_offer_bodies.append(offer_body)
        return pred_actions, pred_offer_bodies


class SingleTargetRecipe:
    name = "single_target"

    def load_training_views(self, *, dataset_dir, stage: str, serialization_version: str = "v1") -> DatasetDict:
        suffix = "_v2" if serialization_version == "v2" else ""
        view = ("single_target_teacher" if stage == "warmstart" else "single_target_main") + suffix
        return load_named_view(dataset_dir=dataset_dir, view_name=view, splits=("train", "valid"))

    def make_train_config(self, *, init_model: str, stage: str, max_source_length: int, max_target_length: int, action_loss_weight: float, offer_loss_weight: float, reset_action_head_on_load: bool, serialization_version: str = "v1") -> TrainRecipeConfig:
        return TrainRecipeConfig(init_model=init_model, stage=stage, max_source_length=max_source_length, max_target_length=max_target_length, serialization_version=serialization_version)

    def build_model_and_tokenizer(self, *, model_source: str, cfg: TrainRecipeConfig):
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        tokenizer.truncation_side = "left"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
        return model, tokenizer

    def build_collator(self, *, tokenizer, model):
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    def tokenize_views(self, *, raw_ds: DatasetDict, tokenizer, cfg: TrainRecipeConfig) -> DatasetDict:
        target_col = "target_text"
        if target_col not in raw_ds["train"].column_names:
            target_candidates = [c for c in ("teacher_target_text", "main_target_text") if c in raw_ds["train"].column_names]
            if not target_candidates:
                raise KeyError("Could not resolve single-target target column")
            target_col = target_candidates[0]

        def _tok(batch):
            model_inputs = tokenizer(batch["source_text"], max_length=cfg.max_source_length, truncation=True)
            labels = tokenizer(text_target=batch[target_col], max_length=cfg.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return raw_ds.map(_tok, batched=True, remove_columns=raw_ds["train"].column_names, load_from_cache_file=True, keep_in_memory=False)

    def build_predictor(self, *, model_dir, device: str = "auto"):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return SingleTargetPredictor(model=model, tokenizer=tokenizer, device=device)

    def build_predictor_from_model(self, *, model, tokenizer):
        device = str(next(model.parameters()).device)
        return SingleTargetPredictor(model=model, tokenizer=tokenizer, device=device)

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from datasets import DatasetDict


@dataclass
class TrainRecipeConfig:
    init_model: str
    stage: str
    max_source_length: int
    max_target_length: int
    action_loss_weight: float = 1.0
    offer_loss_weight: float = 1.0
    reset_action_head_on_load: bool = False
    serialization_version: str = "v1"


class Recipe(Protocol):
    name: str
    def load_training_views(self, *, dataset_dir, stage: str, serialization_version: str = "v1") -> DatasetDict: ...
    def make_train_config(self, *, init_model: str, stage: str, max_source_length: int, max_target_length: int, action_loss_weight: float, offer_loss_weight: float, reset_action_head_on_load: bool, serialization_version: str = "v1") -> TrainRecipeConfig: ...
    def build_model_and_tokenizer(self, *, model_source: str, cfg: TrainRecipeConfig): ...
    def build_collator(self, *, tokenizer, model): ...
    def tokenize_views(self, *, raw_ds: DatasetDict, tokenizer, cfg: TrainRecipeConfig) -> DatasetDict: ...
    def build_predictor(self, *, model_dir, device: str = "auto"): ...
    def build_predictor_from_model(self, *, model, tokenizer): ...

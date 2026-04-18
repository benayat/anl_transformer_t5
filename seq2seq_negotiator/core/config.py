
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

DATASET_ENV_VAR = "SEQNEG_DATASET_DIR"

def resolve_dataset_dir(dataset_dir: str | Path | None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir)
    env_value = os.getenv(DATASET_ENV_VAR)
    if env_value:
        return Path(env_value)
    raise ValueError(f"Dataset directory not provided. Pass --dataset-dir or set {DATASET_ENV_VAR}.")

@dataclass
class BatchPlan:
    world_size: int
    requested_global_batch: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_global_batch: int

def build_batch_plan(requested_global_batch: int, *, world_size: int, max_auto_per_device_batch_size: int) -> BatchPlan:
    if requested_global_batch <= 0:
        raise ValueError("requested_global_batch must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if max_auto_per_device_batch_size <= 0:
        raise ValueError("max_auto_per_device_batch_size must be > 0")
    base = max(1, requested_global_batch // max(1, world_size))
    per_device = max(1, min(max_auto_per_device_batch_size, base))
    local_global = per_device * world_size
    grad_acc = max(1, (requested_global_batch + local_global - 1) // local_global)
    effective = per_device * world_size * grad_acc
    return BatchPlan(world_size, requested_global_batch, per_device, grad_acc, effective)

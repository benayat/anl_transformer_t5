
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from .checkpointing import resolve_resume_checkpoint
from .config import BatchPlan, build_batch_plan
from .ddp import get_world_size, is_main_process
from .signals import install_signal_handlers, stop_reason, stop_requested

class SignalSaveCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if not stop_requested():
            return control
        if self.trainer is not None and is_main_process():
            print(f"[seqneg] stop requested ({stop_reason()}); requesting checkpoint save at step {state.global_step}")
            control.should_save = True
        control.should_training_stop = True
        return control

class LossBreakdownTrainer(Seq2SeqTrainer):
    def __init__(self, *args, debug_first_n_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_first_n_steps = debug_first_n_steps
        self._latest_action_loss: float | None = None
        self._latest_offer_loss: float | None = None

    @staticmethod
    def _to_float(x):
        if x is None:
            return None
        return float(x.detach().cpu())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model returned loss=None")
        self._latest_action_loss = self._to_float(getattr(outputs, "action_loss", None))
        self._latest_offer_loss = self._to_float(getattr(outputs, "offer_loss", None))
        if self.state.global_step < self.debug_first_n_steps:
            logs = {
                "debug_step": int(self.state.global_step),
                "loss": self._to_float(loss),
                "action_loss": self._latest_action_loss,
                "offer_loss": self._latest_offer_loss,
            }
            if "action_labels" in inputs:
                labels = inputs["action_labels"]
                logs["n_accept"] = int((labels == 0).sum().item())
                logs["n_offer"] = int((labels == 1).sum().item())
            print(logs)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite total loss. loss={self._to_float(loss)} "
                f"action_loss={self._latest_action_loss} offer_loss={self._latest_offer_loss}"
            )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if self._latest_action_loss is not None:
            logs["action_loss"] = round(self._latest_action_loss, 6)
        if self._latest_offer_loss is not None:
            logs["offer_loss"] = round(self._latest_offer_loss, 6)
        return super().log(logs, start_time=start_time)

@dataclass
class CommonTrainArgs:
    output_dir: Path
    epochs: float
    lr: float
    batch_size: int
    eval_batch_size: int
    max_auto_per_device_batch_size: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    max_source_length: int
    max_target_length: int
    debug_first_n_steps: int
    resume: str
    resume_from_checkpoint: Optional[Path]

def build_training_arguments(cfg: CommonTrainArgs, *, report_to: list[str] | None = None) -> tuple[Seq2SeqTrainingArguments, BatchPlan, BatchPlan]:
    world_size = get_world_size()
    train_plan = build_batch_plan(cfg.batch_size, world_size=world_size, max_auto_per_device_batch_size=cfg.max_auto_per_device_batch_size)
    eval_plan = build_batch_plan(cfg.eval_batch_size, world_size=world_size, max_auto_per_device_batch_size=cfg.max_auto_per_device_batch_size)
    args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.lr,
        per_device_train_batch_size=train_plan.per_device_batch_size,
        per_device_eval_batch_size=eval_plan.per_device_batch_size,
        gradient_accumulation_steps=train_plan.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if cfg.eval_strategy == "steps" else None,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps if cfg.save_strategy == "steps" else None,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=False,
        report_to=report_to or [],
        remove_unused_columns=False,
        logging_nan_inf_filter=False,
        ddp_find_unused_parameters=False,
    )
    return args, train_plan, eval_plan

def install_runtime_callbacks(*, trainer: Seq2SeqTrainer, extra_callbacks: list[TrainerCallback] | None = None) -> None:
    install_signal_handlers()
    signal_cb = SignalSaveCallback()
    signal_cb.trainer = trainer
    trainer.add_callback(signal_cb)
    for cb in extra_callbacks or []:
        if hasattr(cb, "trainer"):
            cb.trainer = trainer
        trainer.add_callback(cb)

def run_training(*, trainer: Seq2SeqTrainer, output_dir: Path, resume: str, resume_from_checkpoint: Optional[Path]):
    resume_path = resolve_resume_checkpoint(output_dir=output_dir, resume=resume, resume_from_checkpoint=resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_path)

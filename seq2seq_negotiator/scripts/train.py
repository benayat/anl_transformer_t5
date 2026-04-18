#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch
import typer

from seq2seq_negotiator.core.config import resolve_dataset_dir
from seq2seq_negotiator.core.token_cache import prepare_or_load_tokenized_dataset
from seq2seq_negotiator.core.trainer_runtime import CommonTrainArgs, LossBreakdownTrainer, build_training_arguments, install_runtime_callbacks, run_training
from seq2seq_negotiator.core.validation import DecodedValidationCallback, DecodedValidationConfig
from seq2seq_negotiator.recipes import RECIPE_REGISTRY

app = typer.Typer(add_completion=False)


def resolve_model_source(*, stage: str, init_model: Optional[str], warmstart_dir: Optional[Path]) -> str:
    if stage == "warmstart":
        if not init_model:
            raise typer.BadParameter("--init-model is required for warmstart")
        return init_model
    if init_model and warmstart_dir:
        raise typer.BadParameter("Provide only one of --init-model or --warmstart-dir for stage=main")
    if not init_model and not warmstart_dir:
        raise typer.BadParameter("Provide one of --init-model or --warmstart-dir for stage=main")
    if init_model:
        return init_model
    assert warmstart_dir is not None
    return str(warmstart_dir)


@app.command()
def main(
    recipe: str = typer.Option(...),
    stage: str = typer.Option(...),
    dataset_dir: Optional[Path] = typer.Option(None),
    output_dir: Path = typer.Option(...),
    init_model: Optional[str] = typer.Option(None),
    warmstart_dir: Optional[Path] = typer.Option(None),
    epochs: float = typer.Option(1.0),
    batch_size: int = typer.Option(128),
    eval_batch_size: int = typer.Option(128),
    max_auto_per_device_batch_size: int = typer.Option(32),
    lr: float = typer.Option(8e-4),
    max_source_length: int = typer.Option(512),
    max_target_length: int = typer.Option(64),
    action_loss_weight: float = typer.Option(1.0),
    offer_loss_weight: float = typer.Option(1.0),
    serialization_version: str = typer.Option("v1"),
    save_strategy: str = typer.Option("steps"),
    save_steps: int = typer.Option(500),
    save_total_limit: int = typer.Option(2),
    logging_steps: int = typer.Option(10),
    eval_strategy: str = typer.Option("steps"),
    eval_steps: int = typer.Option(2000),
    debug_first_n_steps: int = typer.Option(0),
    resume: str = typer.Option("auto"),
    resume_from_checkpoint: Optional[Path] = typer.Option(None),
    validation_log_steps: int = typer.Option(0),
    validation_log_max_rows: int = typer.Option(2048),
    validation_log_batch_size: int = typer.Option(64),
    validation_log_num_beams: int = typer.Option(1),
) -> None:
    if recipe not in RECIPE_REGISTRY:
        raise typer.BadParameter(f"Unknown recipe: {recipe}")
    if stage not in {"warmstart", "main"}:
        raise typer.BadParameter("stage must be warmstart or main")
    if serialization_version not in {"v1", "v2"}:
        raise typer.BadParameter("serialization_version must be v1 or v2")

    ds_dir = resolve_dataset_dir(dataset_dir)
    recipe_impl = RECIPE_REGISTRY[recipe]
    model_source = resolve_model_source(stage=stage, init_model=init_model, warmstart_dir=warmstart_dir)
    raw_ds = recipe_impl.load_training_views(dataset_dir=ds_dir, stage=stage, serialization_version=serialization_version)

    cfg = recipe_impl.make_train_config(
        init_model=model_source,
        stage=stage,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        action_loss_weight=action_loss_weight,
        offer_loss_weight=offer_loss_weight,
        reset_action_head_on_load=(stage == "warmstart" or (stage == "main" and init_model is not None and recipe == "multitask")),
        serialization_version=serialization_version,
    )
    model, tokenizer = recipe_impl.build_model_and_tokenizer(model_source=model_source, cfg=cfg)
    collator = recipe_impl.build_collator(tokenizer=tokenizer, model=model)

    tokenized = prepare_or_load_tokenized_dataset(
        cache_root=output_dir / "token_cache",
        cache_key_payload={
            "recipe": recipe,
            "stage": stage,
            "model_source": model_source,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "dataset_dir": str(ds_dir.resolve()),
            "serialization_version": serialization_version,
        },
        build_fn=lambda: recipe_impl.tokenize_views(raw_ds=raw_ds, tokenizer=tokenizer, cfg=cfg),
    )

    if hasattr(model, "debug_forward_probe"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        sample_count = min(4, len(tokenized["train"]))
        if sample_count:
            sample_features = [tokenized["train"][i] for i in range(sample_count)]
            batch = collator(sample_features)
            batch = {k: v.to(device) for k, v in batch.items()}
            print("=== PRE-FLIGHT PROBE ===")
            print(model.debug_forward_probe(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]))
            print("=== END PRE-FLIGHT PROBE ===")

    common = CommonTrainArgs(
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_auto_per_device_batch_size=max_auto_per_device_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        debug_first_n_steps=debug_first_n_steps,
        resume=resume,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    training_args, train_plan, eval_plan = build_training_arguments(common)
    print({"recipe": recipe, "stage": stage, "dataset_dir": str(ds_dir), "model_source": model_source, "serialization_version": serialization_version, "train_batch_plan": train_plan.__dict__, "eval_batch_plan": eval_plan.__dict__})

    trainer = LossBreakdownTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        processing_class=tokenizer,
        data_collator=collator,
        debug_first_n_steps=debug_first_n_steps,
    )

    callbacks = []
    if validation_log_steps > 0:
        predictor = recipe_impl.build_predictor_from_model(model=model, tokenizer=tokenizer)
        callbacks.append(DecodedValidationCallback(
            predictor=predictor,
            valid_dataset=raw_ds["valid"],
            config=DecodedValidationConfig(
                steps=validation_log_steps,
                max_rows=validation_log_max_rows,
                batch_size=validation_log_batch_size,
                num_beams=validation_log_num_beams,
                max_source_length=max_source_length,
                max_new_tokens=max_target_length,
            ),
        ))

    install_runtime_callbacks(trainer=trainer, extra_callbacks=callbacks)
    run_training(trainer=trainer, output_dir=output_dir, resume=resume, resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[seqneg] model saved to {output_dir}")


if __name__ == "__main__":
    app()

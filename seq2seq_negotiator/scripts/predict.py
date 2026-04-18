#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import typer

from seq2seq_negotiator.core.metrics import parse_issue_vocab_from_source, parse_offer_body
from seq2seq_negotiator.recipes import RECIPE_REGISTRY

app = typer.Typer(add_completion=False)


@app.command()
def main(
    recipe: str = typer.Option(...),
    model_dir: Path = typer.Option(..., exists=True, file_okay=False),
    dataset_path: Path = typer.Option(..., exists=True, dir_okay=False),
    row_index: int = typer.Option(0),
    max_source_length: int = typer.Option(512),
    max_new_tokens: int = typer.Option(64),
    num_beams: int = typer.Option(1),
    device: str = typer.Option("auto"),
) -> None:
    if recipe not in RECIPE_REGISTRY:
        raise typer.BadParameter(f"Unknown recipe: {recipe}")
    df = pd.read_parquet(dataset_path)
    if row_index < 0 or row_index >= len(df):
        raise typer.BadParameter(f"row_index out of range: {row_index}")
    row = df.iloc[row_index]
    predictor = RECIPE_REGISTRY[recipe].build_predictor(model_dir=model_dir, device=device)
    pred_actions, pred_offer_bodies = predictor.predict_batch([row["source_text"]], max_source_length=max_source_length, max_new_tokens=max_new_tokens, num_beams=num_beams)
    issue_vocab = parse_issue_vocab_from_source(row["source_text"])
    parsed_pred_bid = parse_offer_body(pred_offer_bodies[0], issue_vocab) if pred_actions[0] == "OFFER" else None

    result = {
        "row_index": row_index,
        "source_text": row["source_text"],
        "pred_action": pred_actions[0],
        "pred_offer_body": pred_offer_bodies[0],
        "parsed_pred_bid": parsed_pred_bid,
    }

    for col in ["teacher_target_text", "main_target_text", "teacher_action_label", "main_action_label", "teacher_offer_target_text", "main_offer_target_text", "target_mode"]:
        if col in row.index:
            result[col] = row[col]

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()

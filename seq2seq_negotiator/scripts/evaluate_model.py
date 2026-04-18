#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import pandas as pd
import typer

from seq2seq_negotiator.core.metrics import (
    compute_binary_f1,
    parse_issue_vocab_from_source,
    parse_offer_body,
    parse_single_target_action_text,
    safe_div,
)
from seq2seq_negotiator.recipes import RECIPE_REGISTRY

app = typer.Typer(add_completion=False)


def summarize_rows(rows_df: pd.DataFrame) -> dict:
    n = len(rows_df)
    gold_offer_mask = rows_df["gold_action"] == "OFFER"
    gold_accept_mask = rows_df["gold_action"] == "ACCEPT"
    pred_accept_mask = rows_df["pred_action"] == "ACCEPT"
    pred_offer_mask = rows_df["pred_action"] == "OFFER"
    tp_accept = int((gold_accept_mask & pred_accept_mask).sum())
    fp_accept = int((~gold_accept_mask & pred_accept_mask).sum())
    fn_accept = int((gold_accept_mask & ~pred_accept_mask).sum())
    summary = {
        "n_rows": int(n),
        "format_valid_rate": safe_div(int(rows_df["pred_format_valid"].sum()), n),
        "action_accuracy": safe_div(int(rows_df["action_correct"].sum()), n),
        "normalized_exact_match": safe_div(int(rows_df["normalized_exact_match"].sum()), n),
        "accept_precision": safe_div(tp_accept, tp_accept + fp_accept),
        "accept_recall": safe_div(tp_accept, tp_accept + fn_accept),
        "accept_f1": compute_binary_f1(tp_accept, fp_accept, fn_accept),
        "gold_accept_rate": safe_div(int(gold_accept_mask.sum()), n),
        "pred_accept_rate": safe_div(int(pred_accept_mask.sum()), n),
        "gold_offer_rate": safe_div(int(gold_offer_mask.sum()), n),
        "pred_offer_rate": safe_div(int(pred_offer_mask.sum()), n),
    }
    gold_offer_rows = rows_df[gold_offer_mask].copy()
    if len(gold_offer_rows):
        summary["offer_bid_exact_match_on_gold_offer"] = float(gold_offer_rows["offer_bid_exact_match"].mean())
        summary["offer_issue_accuracy_on_gold_offer"] = float(gold_offer_rows["offer_issue_accuracy"].mean())
    else:
        summary["offer_bid_exact_match_on_gold_offer"] = math.nan
        summary["offer_issue_accuracy_on_gold_offer"] = math.nan

    both_offer_rows = rows_df[gold_offer_mask & pred_offer_mask].copy()
    if len(both_offer_rows):
        summary["offer_bid_exact_match_when_both_offer"] = float(both_offer_rows["offer_bid_exact_match"].mean())
        summary["offer_issue_accuracy_when_both_offer"] = float(both_offer_rows["offer_issue_accuracy"].mean())
    else:
        summary["offer_bid_exact_match_when_both_offer"] = math.nan
        summary["offer_issue_accuracy_when_both_offer"] = math.nan
    return summary


def normalize_single_target_gold(text: str) -> tuple[str, str]:
    action, offer_body = parse_single_target_action_text(text)
    return action, offer_body


@app.command()
def main(
    recipe: str = typer.Option(...),
    stage: str = typer.Option("main"),
    model_dir: Path = typer.Option(..., exists=True, file_okay=False),
    dataset_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(32),
    max_source_length: int = typer.Option(512),
    max_new_tokens: int = typer.Option(64),
    num_beams: int = typer.Option(1),
    device: str = typer.Option("auto"),
) -> None:
    if recipe not in RECIPE_REGISTRY:
        raise typer.BadParameter(f"Unknown recipe: {recipe}")
    if stage not in {"warmstart", "main"}:
        raise typer.BadParameter("stage must be warmstart or main")

    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_impl = RECIPE_REGISTRY[recipe]
    predictor = recipe_impl.build_predictor(model_dir=model_dir, device=device)
    df = pd.read_parquet(dataset_path)
    rows = []
    source_texts = df["source_text"].tolist()

    if recipe == "single_target":
        target_col = "teacher_target_text" if stage == "warmstart" else "main_target_text"
        parsed = [normalize_single_target_gold(str(x)) for x in df[target_col].tolist()]
        gold_actions = [a for a, _ in parsed]
        gold_offer_targets = [b for _, b in parsed]
    else:
        action_col = "teacher_action_label" if stage == "warmstart" else "main_action_label"
        offer_col = "teacher_offer_target_text" if stage == "warmstart" else "main_offer_target_text"
        gold_actions = [str(x).upper() for x in df[action_col].tolist()]
        gold_offer_targets = [str(x or "") for x in df[offer_col].tolist()]

    for start in range(0, len(source_texts), batch_size):
        batch_source_texts = source_texts[start : start + batch_size]
        pred_actions, pred_offer_bodies = predictor.predict_batch(
            batch_source_texts,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        for i, source_text in enumerate(batch_source_texts):
            gold_action = gold_actions[start + i]
            pred_action = pred_actions[i]
            gold_offer_body = gold_offer_targets[start + i]
            pred_offer_body = pred_offer_bodies[i]
            issue_vocab = parse_issue_vocab_from_source(source_text)

            pred_format_valid = pred_action == "ACCEPT"
            offer_bid_exact_match = math.nan
            offer_issue_accuracy = math.nan
            normalized_exact_match = False
            gold_bid_json = None
            pred_bid_json = None

            if gold_action == "ACCEPT":
                normalized_exact_match = pred_action == "ACCEPT"
            else:
                gold_bid = parse_offer_body(gold_offer_body, issue_vocab)
                gold_bid_json = json.dumps(gold_bid, ensure_ascii=False, sort_keys=True)
                if pred_action != "OFFER":
                    offer_bid_exact_match = 0.0
                    offer_issue_accuracy = 0.0
                    normalized_exact_match = False
                else:
                    pred_bid = parse_offer_body(pred_offer_body, issue_vocab)
                    pred_bid_json = json.dumps(pred_bid, ensure_ascii=False, sort_keys=True)
                    pred_format_valid = bool(pred_bid) and set(pred_bid.keys()) == set(gold_bid.keys())
                    offer_bid_exact_match = float(pred_bid == gold_bid)
                    if gold_bid:
                        per_issue = [float(pred_bid.get(issue) == gold_val) for issue, gold_val in gold_bid.items()]
                        offer_issue_accuracy = float(sum(per_issue) / len(per_issue))
                    else:
                        offer_issue_accuracy = math.nan
                    normalized_exact_match = bool(pred_bid == gold_bid)

            rows.append({
                "row_index": int(start + i),
                "source_text": source_text,
                "gold_action": gold_action,
                "pred_action": pred_action,
                "action_correct": bool(gold_action == pred_action),
                "pred_format_valid": bool(pred_format_valid),
                "normalized_exact_match": bool(normalized_exact_match),
                "gold_offer_body": gold_offer_body,
                "pred_offer_body": pred_offer_body,
                "gold_bid_json": gold_bid_json,
                "pred_bid_json": pred_bid_json,
                "offer_bid_exact_match": offer_bid_exact_match,
                "offer_issue_accuracy": offer_issue_accuracy,
            })

    rows_df = pd.DataFrame(rows)
    summary = summarize_rows(rows_df)
    rows_df.to_parquet(output_dir / "evaluation_rows.parquet", index=False)
    rows_df.to_csv(output_dir / "evaluation_rows.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()

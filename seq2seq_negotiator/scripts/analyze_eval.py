
#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

def mean_or_nan(series):
    if len(series) == 0:
        return math.nan
    return float(series.mean())

@app.command()
def main(
    eval_path: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    top_n: int = typer.Option(50),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(eval_path)

    by_gold_action = {}
    for action, sub in df.groupby("gold_action"):
        by_gold_action[str(action)] = {
            "n_rows": int(len(sub)),
            "action_accuracy": mean_or_nan(sub["action_correct"]),
            "offer_bid_exact_match": mean_or_nan(sub["offer_bid_exact_match"].dropna()),
            "offer_issue_accuracy": mean_or_nan(sub["offer_issue_accuracy"].dropna()),
        }

    failures = df[df["action_correct"] == False].copy().head(top_n)
    offers_wrong = df[(df["gold_action"] == "OFFER") & (df["offer_bid_exact_match"] == 0.0)].copy().head(top_n)
    summary = {
        "n_rows": int(len(df)),
        "by_gold_action": by_gold_action,
        "n_action_failures": int((~df["action_correct"]).sum()),
        "n_offer_wrong": int(((df["gold_action"] == "OFFER") & (df["offer_bid_exact_match"] == 0.0)).sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    failures.to_csv(output_dir / "top_action_failures.csv", index=False)
    offers_wrong.to_csv(output_dir / "top_offer_failures.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()

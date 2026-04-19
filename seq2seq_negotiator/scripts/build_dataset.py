
#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer

app = typer.Typer(add_completion=False)


@dataclass
class BuilderConfig:
    max_history_turns: int = 64
    train_frac: float = 0.90
    valid_frac: float = 0.05
    test_frac: float = 0.05
    seed: int = 13
    include_final_metrics_in_source: bool = False
    hindsight_accept_utility_epsilon: float = 0.0
    hindsight_accept_use_final_utility: bool = True
    hindsight_accept_use_next_self_offer_utility: bool = True
    hindsight_accept_use_legacy_final_agreement_fallback: bool = True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed parsing JSONL line {line_no} from {path}: {e}") from e
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


# ----------------------------------------------------------------------
# Bid normalization helpers
# ----------------------------------------------------------------------
def _canonical_issue_order(issue_names: list[str], bid: Any) -> list[str]:
    if issue_names:
        return list(issue_names)
    if isinstance(bid, dict):
        return sorted(str(k) for k in bid.keys())
    return []


def bid_to_ordered_values(issue_names: list[str], bid: Any) -> list[Any] | None:
    """
    Normalize a bid into ordered values according to issue_names.

    Supports:
    - None
    - list / tuple
    - dict keyed by issue names
    """
    if bid is None:
        return None

    if isinstance(bid, (list, tuple)):
        return list(bid)

    if isinstance(bid, dict):
        order = _canonical_issue_order(issue_names, bid)
        values: list[Any] = []
        for issue in order:
            values.append(bid.get(issue))
        return values

    raise TypeError(f"Unsupported bid type: {type(bid)!r}")


def canonicalize_bid(issue_names: list[str], bid: Any) -> tuple[Any, ...] | None:
    values = bid_to_ordered_values(issue_names, bid)
    if values is None:
        return None
    return tuple(values)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def find_last_visible_opp_offer(prefix_turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in reversed(prefix_turns):
        if turn.get("actor") == "opp" and turn.get("action_type") == "offer":
            return turn
    return None


def accept_target_text_for(target_text_fn) -> str:
    return "A" if target_text_fn is serialize_target_v2 else "ACTION ACCEPT"


# ----------------------------------------------------------------------
# v1 serialization (kept for compatibility)
# ----------------------------------------------------------------------
def serialize_bid_v1(issue_names: list[str], bid: Any) -> str:
    ordered = bid_to_ordered_values(issue_names, bid)
    if ordered is None:
        return "NONE"

    return " | ".join(
        f"{issue_names[i] if i < len(issue_names) else f'issue_{i}'}={value}"
        for i, value in enumerate(ordered)
    )


def serialize_history_v1(issue_names: list[str], turns: list[dict[str, Any]], max_history_turns: int) -> list[str]:
    lines: list[str] = []
    for t in turns[-max_history_turns:]:
        actor = t["actor"]
        action_type = t["action_type"]
        rel_time = t.get("rel_time")
        step = t.get("step")
        self_utility = t.get("self_utility")
        opp_utility_est = t.get("opp_utility_est")

        prefix = f"TURN actor={actor} action={action_type}"
        if step is not None:
            prefix += f" step={step}"
        if rel_time is not None:
            prefix += f" rel_time={float(rel_time):.6f}"
        prefix += f" bid={serialize_bid_v1(issue_names, t.get('bid'))}"
        if self_utility is not None:
            prefix += f" self_utility={float(self_utility):.6f}"
        if opp_utility_est is not None:
            prefix += f" opp_utility_est={float(opp_utility_est):.6f}"
        lines.append(prefix)
    return lines


def serialize_source_v1(row: dict[str, Any], prefix_turns: list[dict[str, Any]], cfg: BuilderConfig) -> str:
    issue_names = list(row.get("issue_names", []))
    issue_values = list(row.get("issue_values", []))

    lines: list[str] = []
    lines.append("<SCENARIO>")
    lines.append(f"SCENARIO_NAME {row.get('scenario_name', 'unknown')}")
    lines.append(f"NEGOTIATION_ID {row.get('negotiation_id', 'unknown')}")
    lines.append(f"FIRST_MOVER {row.get('first_mover', 'unknown')}")

    for i, values in enumerate(issue_values):
        issue = issue_names[i] if i < len(issue_names) else f"issue_{i}"
        lines.append(f"ISSUE {issue} VALUES {','.join(map(str, values))}")

    lines.append("<STATE>")
    lines.append(f"RESERVED_VALUE {float(row.get('reserved_value', 0.0)):.6f}")
    lines.append(f"MAX_UTILITY {float(row.get('max_utility', 1.0)):.6f}")

    if prefix_turns:
        last_turn = prefix_turns[-1]
        lines.append(f"CURRENT_STEP {int(last_turn.get('step', len(prefix_turns)-1))}")
        lines.append(f"CURRENT_REL_TIME {float(last_turn.get('rel_time', 0.0)):.6f}")

        current_opp_offer = None
        current_opp_offer_utility = None
        for prev in reversed(prefix_turns):
            if prev.get("actor") == "opp" and prev.get("action_type") == "offer":
                current_opp_offer = prev.get("bid")
                current_opp_offer_utility = prev.get("self_utility")
                break

        if current_opp_offer is None:
            lines.append("CURRENT_OPPONENT_OFFER NONE")
        else:
            lines.append(f"CURRENT_OPPONENT_OFFER {serialize_bid_v1(issue_names, current_opp_offer)}")
            if current_opp_offer_utility is not None:
                lines.append(f"CURRENT_OPPONENT_OFFER_UTILITY {float(current_opp_offer_utility):.6f}")

        opp_utils = [
            float(t["self_utility"])
            for t in prefix_turns
            if t.get("actor") == "opp" and t.get("action_type") == "offer" and t.get("self_utility") is not None
        ]
        if opp_utils:
            lines.append(f"BEST_SEEN_OPPONENT_UTILITY {max(opp_utils):.6f}")
    else:
        lines.append("CURRENT_STEP 0")
        lines.append("CURRENT_REL_TIME 0.000000")
        lines.append("CURRENT_OPPONENT_OFFER NONE")

    if cfg.include_final_metrics_in_source:
        for key in ("final_utility", "final_advantage", "final_deception", "final_score"):
            if row.get(key) is not None:
                lines.append(f"{key.upper()} {float(row[key]):.6f}")

    lines.append("<HISTORY>")
    lines.extend(serialize_history_v1(issue_names, prefix_turns, cfg.max_history_turns))
    return "\n".join(lines)


def serialize_target_v1(row: dict[str, Any], turn: dict[str, Any]) -> str:
    action_type = turn["action_type"]
    if action_type == "accept":
        return "ACTION ACCEPT"
    if action_type != "offer":
        raise ValueError(f"Unsupported action_type for target: {action_type}")

    issue_names = list(row.get("issue_names", []))
    ordered = bid_to_ordered_values(issue_names, turn.get("bid"))
    if ordered is None:
        raise ValueError("Offer target is missing bid")

    lines = ["ACTION OFFER"]
    for i, value in enumerate(ordered):
        issue = issue_names[i] if i < len(issue_names) else f"issue_{i}"
        lines.append(f"{issue} = {value}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# v2 serialization
# ----------------------------------------------------------------------
def quantize_01(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except Exception:
        return "NA"
    x = max(0.0, min(1.0, x))
    return str(int(round(x * 1000.0)))


def build_anon_maps(row: dict[str, Any]) -> dict[str, Any]:
    issue_names = list(row.get("issue_names", []))
    issue_values = list(row.get("issue_values", []))
    anon_issue_names = [f"i{i+1}" for i in range(len(issue_names))]
    to_anon_issue = {real: anon for real, anon in zip(issue_names, anon_issue_names)}
    to_anon_val: dict[str, dict[Any, str]] = {}
    for real_issue, values in zip(issue_names, issue_values):
        to_anon_val[real_issue] = {value: f"v{j+1}" for j, value in enumerate(values)}
    return {
        "issue_names": issue_names,
        "anon_issue_names": anon_issue_names,
        "to_anon_issue": to_anon_issue,
        "to_anon_val": to_anon_val,
        "issue_values": issue_values,
    }


def to_compact_bid(row: dict[str, Any], maps: dict[str, Any], bid: Any) -> str:
    ordered = bid_to_ordered_values(maps["issue_names"], bid)
    if ordered is None:
        return "N"

    issue_names = maps["issue_names"]
    to_anon_val = maps["to_anon_val"]
    compact: list[str] = []
    for i, value in enumerate(ordered):
        if i >= len(issue_names):
            break
        issue = issue_names[i]
        compact.append(to_anon_val.get(issue, {}).get(value, str(value)))
    return ",".join(compact) if compact else "N"


def serialize_source_v2(row: dict[str, Any], prefix_turns: list[dict[str, Any]], cfg: BuilderConfig) -> str:
    maps = build_anon_maps(row)
    issue_names = maps["issue_names"]
    anon_issue_names = maps["anon_issue_names"]
    issue_values = maps["issue_values"]

    first = str(row.get("first_mover", "unknown")).lower()
    first_tag = "S" if first == "self" else "O" if first == "opp" else "X"
    scenario = str(row.get("scenario_name", "unknown")).replace(" ", "_")

    best_seen_opp_utility = None
    current_opp_offer = None
    current_opp_offer_utility = None
    last_self_offer = None
    step = 0
    rel_time = 0.0

    if prefix_turns:
        last_turn = prefix_turns[-1]
        step = int(last_turn.get("step", len(prefix_turns) - 1))
        rel_time = float(last_turn.get("rel_time", 0.0))
        for prev in reversed(prefix_turns):
            if current_opp_offer is None and prev.get("actor") == "opp" and prev.get("action_type") == "offer":
                current_opp_offer = prev.get("bid")
                current_opp_offer_utility = prev.get("self_utility")
            if last_self_offer is None and prev.get("actor") == "self" and prev.get("action_type") == "offer":
                last_self_offer = prev.get("bid")
            if current_opp_offer is not None and last_self_offer is not None:
                break
        opp_utils = [
            float(t["self_utility"])
            for t in prefix_turns
            if t.get("actor") == "opp" and t.get("action_type") == "offer" and t.get("self_utility") is not None
        ]
        if opp_utils:
            best_seen_opp_utility = max(opp_utils)

    lines: list[str] = []
    lines.append(f"@S {scenario} {len(issue_names)} {first_tag}")
    for anon_issue, values in zip(anon_issue_names, issue_values):
        value_ids = [f"v{j+1}" for j in range(len(values))]
        lines.append(f"@V {anon_issue}:{'|'.join(value_ids)}")

    lines.append(
        "@C "
        f"s={step} "
        f"t={quantize_01(rel_time)} "
        f"r={quantize_01(row.get('reserved_value'))} "
        f"b={quantize_01(best_seen_opp_utility)} "
        f"co={to_compact_bid(row, maps, current_opp_offer)} "
        f"cu={quantize_01(current_opp_offer_utility)} "
        f"ls={to_compact_bid(row, maps, last_self_offer)}"
    )

    lines.append("@H")
    for t in prefix_turns[-cfg.max_history_turns:]:
        actor = "S" if t.get("actor") == "self" else "O"
        action = "O" if t.get("action_type") == "offer" else "A"
        if action == "A":
            lines.append(f"{actor}{action}")
        else:
            lines.append(
                f"{actor}{action} {quantize_01(t.get('rel_time'))} {quantize_01(t.get('self_utility'))} {to_compact_bid(row, maps, t.get('bid'))}"
            )
    return "\n".join(lines)


def serialize_target_v2(row: dict[str, Any], turn: dict[str, Any]) -> str:
    action_type = turn["action_type"]
    if action_type == "accept":
        return "A"
    if action_type != "offer":
        raise ValueError(f"Unsupported action_type for target: {action_type}")
    maps = build_anon_maps(row)
    return f"O {to_compact_bid(row, maps, turn.get('bid'))}"


def build_main_target(
    target_text_fn,
    row: dict[str, Any],
    prefix_turns: list[dict[str, Any]],
    next_turn: dict[str, Any],
    cfg: BuilderConfig,
) -> tuple[str, str]:
    teacher = target_text_fn(row, next_turn)
    if next_turn.get("action_type") != "offer" or not prefix_turns:
        return teacher, "teacher_passthrough"

    issue_names = list(row.get("issue_names", []))
    accept_target = accept_target_text_for(target_text_fn)
    epsilon = max(0.0, float(cfg.hindsight_accept_utility_epsilon))

    last_visible_opp_offer = find_last_visible_opp_offer(prefix_turns)
    if last_visible_opp_offer is None:
        return teacher, "teacher_passthrough"

    current_opp_utility = safe_float(last_visible_opp_offer.get("self_utility"))
    next_self_utility = safe_float(next_turn.get("self_utility"))
    final_utility = safe_float(row.get("final_utility"))
    reserved_value = safe_float(row.get("reserved_value"))

    if current_opp_utility is not None:
        if reserved_value is not None and current_opp_utility + epsilon < reserved_value:
            return teacher, "teacher_passthrough"

        if cfg.hindsight_accept_use_final_utility and final_utility is not None:
            if current_opp_utility + epsilon >= final_utility:
                return accept_target, "hindsight_accept_visible_opp_ge_final_utility"

        if cfg.hindsight_accept_use_next_self_offer_utility and next_self_utility is not None:
            if current_opp_utility + epsilon >= next_self_utility:
                return accept_target, "hindsight_accept_visible_opp_ge_next_self_utility"

    if cfg.hindsight_accept_use_legacy_final_agreement_fallback:
        if (
            row.get("final_agreement") is not None
            and canonicalize_bid(issue_names, last_visible_opp_offer.get("bid")) == canonicalize_bid(issue_names, row.get("final_agreement"))
        ):
            return accept_target, "hindsight_accept_final_agreement_visible"

    return teacher, "teacher_passthrough"


def target_text_to_action_label(target_text: str) -> str:
    text = str(target_text).strip().upper()
    if text == "A" or text.startswith("ACTION ACCEPT"):
        return "ACCEPT"
    if text == "O" or text.startswith("ACTION OFFER") or text.startswith("O "):
        return "OFFER"
    raise ValueError(f"Unsupported target text: {target_text!r}")


def target_text_to_offer_body(target_text: str) -> str:
    text = str(target_text).strip()
    upper = text.upper()
    if upper == "A" or upper.startswith("ACTION ACCEPT"):
        return ""
    if upper == "O":
        return ""
    if upper.startswith("ACTION OFFER"):
        return text[len("ACTION OFFER") :].strip()
    if upper.startswith("O "):
        return text[2:].strip()
    raise ValueError(f"Unsupported target text: {target_text!r}")


def build_examples(rows: list[dict[str, Any]], cfg: BuilderConfig, *, source_text_fn, target_text_fn) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        turns = row.get("turns", [])
        if not turns:
            continue

        for idx, turn in enumerate(turns):
            if turn.get("actor") != "self":
                continue
            if turn.get("action_type") not in {"offer", "accept"}:
                continue

            prefix_turns = turns[:idx]
            teacher_target_text = target_text_fn(row, turn)
            main_target_text, target_mode = build_main_target(target_text_fn, row, prefix_turns, turn, cfg)
            source_text = source_text_fn(row, prefix_turns, cfg)

            teacher_action_label = target_text_to_action_label(teacher_target_text)
            main_action_label = target_text_to_action_label(main_target_text)
            last_visible_opp_offer = find_last_visible_opp_offer(prefix_turns)

            examples.append({
                "source_text": source_text,
                "teacher_target_text": teacher_target_text,
                "main_target_text": main_target_text,
                "target_mode": target_mode,
                "teacher_action_label": teacher_action_label,
                "main_action_label": main_action_label,
                "teacher_offer_target_text": target_text_to_offer_body(teacher_target_text),
                "main_offer_target_text": target_text_to_offer_body(main_target_text),
                "teacher_is_offer": int(teacher_action_label == "OFFER"),
                "main_is_offer": int(main_action_label == "OFFER"),
                "negotiation_id": row.get("negotiation_id"),
                "scenario_name": row.get("scenario_name"),
                "turn_index": idx,
                "step": turn.get("step"),
                "rel_time": turn.get("rel_time"),
                "reserved_value": row.get("reserved_value"),
                "current_visible_opp_utility": safe_float(last_visible_opp_offer.get("self_utility")) if last_visible_opp_offer else None,
                "next_self_utility": safe_float(turn.get("self_utility")),
                "max_utility": row.get("max_utility"),
                "final_utility": row.get("final_utility"),
                "final_advantage": row.get("final_advantage"),
                "final_deception": row.get("final_deception"),
                "final_score": row.get("final_score"),
                "n_turns": row.get("n_turns"),
            })

    if not examples:
        raise ValueError("No examples built from input traces")
    return examples


def split_by_negotiation(examples: list[dict[str, Any]], cfg: BuilderConfig):
    ids = sorted({ex["negotiation_id"] for ex in examples})
    rng = random.Random(cfg.seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(round(n * cfg.train_frac)))
    n_valid = max(1, int(round(n * cfg.valid_frac)))
    if n_train + n_valid >= n:
        n_valid = max(1, min(n_valid, n - 2))
        n_train = max(1, n - n_valid - 1)
    n_test = n - n_train - n_valid
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_valid -= 1

    train_ids = set(ids[:n_train])
    valid_ids = set(ids[n_train:n_train + n_valid])
    test_ids = set(ids[n_train + n_valid:])

    train_rows = [ex for ex in examples if ex["negotiation_id"] in train_ids]
    valid_rows = [ex for ex in examples if ex["negotiation_id"] in valid_ids]
    test_rows = [ex for ex in examples if ex["negotiation_id"] in test_ids]
    return pd.DataFrame(train_rows), pd.DataFrame(valid_rows), pd.DataFrame(test_rows)


def write_view(base: Path, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]):
    base.mkdir(parents=True, exist_ok=True)
    train_df.loc[:, columns].to_parquet(base / "train.parquet", index=False)
    valid_df.loc[:, columns].to_parquet(base / "valid.parquet", index=False)
    test_df.loc[:, columns].to_parquet(base / "test.parquet", index=False)


@app.command()
def main(
    input: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to seqneg_traces.jsonl"),
    output_dir: Path = typer.Option(..., help="Output dataset root directory"),
    max_history_turns: int = typer.Option(64),
    train_frac: float = typer.Option(0.90),
    valid_frac: float = typer.Option(0.05),
    test_frac: float = typer.Option(0.05),
    seed: int = typer.Option(13),
    include_final_metrics_in_source: bool = typer.Option(False),
    hindsight_accept_utility_epsilon: float = typer.Option(0.0),
    hindsight_accept_use_final_utility: bool = typer.Option(True),
    hindsight_accept_use_next_self_offer_utility: bool = typer.Option(True),
    hindsight_accept_use_legacy_final_agreement_fallback: bool = typer.Option(True),
):
    cfg = BuilderConfig(
        max_history_turns=max_history_turns,
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
        seed=seed,
        include_final_metrics_in_source=include_final_metrics_in_source,
        hindsight_accept_utility_epsilon=hindsight_accept_utility_epsilon,
        hindsight_accept_use_final_utility=hindsight_accept_use_final_utility,
        hindsight_accept_use_next_self_offer_utility=hindsight_accept_use_next_self_offer_utility,
        hindsight_accept_use_legacy_final_agreement_fallback=hindsight_accept_use_legacy_final_agreement_fallback,
    )
    total = cfg.train_frac + cfg.valid_frac + cfg.test_frac
    if abs(total - 1.0) > 1e-8:
        raise typer.BadParameter(f"train/valid/test fractions must sum to 1.0, got {total}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = load_jsonl(input)

    examples_v1 = build_examples(raw_rows, cfg, source_text_fn=serialize_source_v1, target_text_fn=serialize_target_v1)
    train_v1, valid_v1, test_v1 = split_by_negotiation(examples_v1, cfg)
    rich_cols = list(train_v1.columns)
    write_view(output_dir / "rich", train_v1, valid_v1, test_v1, rich_cols)
    write_view(output_dir / "single_target_teacher", train_v1, valid_v1, test_v1, ["source_text", "teacher_target_text"])
    write_view(output_dir / "single_target_main", train_v1, valid_v1, test_v1, ["source_text", "main_target_text"])
    write_view(output_dir / "multitask_teacher", train_v1, valid_v1, test_v1, ["source_text", "teacher_action_label", "teacher_offer_target_text"])
    write_view(output_dir / "multitask_main", train_v1, valid_v1, test_v1, ["source_text", "main_action_label", "main_offer_target_text"])

    examples_v2 = build_examples(raw_rows, cfg, source_text_fn=serialize_source_v2, target_text_fn=serialize_target_v2)
    train_v2, valid_v2, test_v2 = split_by_negotiation(examples_v2, cfg)
    rich_cols_v2 = list(train_v2.columns)
    write_view(output_dir / "rich_v2", train_v2, valid_v2, test_v2, rich_cols_v2)
    write_view(output_dir / "single_target_teacher_v2", train_v2, valid_v2, test_v2, ["source_text", "teacher_target_text"])
    write_view(output_dir / "single_target_main_v2", train_v2, valid_v2, test_v2, ["source_text", "main_target_text"])
    write_view(output_dir / "multitask_teacher_v2", train_v2, valid_v2, test_v2, ["source_text", "teacher_action_label", "teacher_offer_target_text"])
    write_view(output_dir / "multitask_main_v2", train_v2, valid_v2, test_v2, ["source_text", "main_action_label", "main_offer_target_text"])

    meta = {
        "input": str(input),
        "n_negotiations": len(raw_rows),
        "n_examples": len(examples_v1),
        "n_examples_v2": len(examples_v2),
        "train_examples": int(len(train_v1)),
        "valid_examples": int(len(valid_v1)),
        "test_examples": int(len(test_v1)),
        "max_history_turns": cfg.max_history_turns,
        "train_frac": cfg.train_frac,
        "valid_frac": cfg.valid_frac,
        "test_frac": cfg.test_frac,
        "seed": cfg.seed,
        "include_final_metrics_in_source": cfg.include_final_metrics_in_source,
        "hindsight_accept_utility_epsilon": cfg.hindsight_accept_utility_epsilon,
        "hindsight_accept_use_final_utility": cfg.hindsight_accept_use_final_utility,
        "hindsight_accept_use_next_self_offer_utility": cfg.hindsight_accept_use_next_self_offer_utility,
        "hindsight_accept_use_legacy_final_agreement_fallback": cfg.hindsight_accept_use_legacy_final_agreement_fallback,
        "views": [
            "rich",
            "single_target_teacher",
            "single_target_main",
            "multitask_teacher",
            "multitask_main",
            "rich_v2",
            "single_target_teacher_v2",
            "single_target_main_v2",
            "multitask_teacher_v2",
            "multitask_main_v2",
        ],
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    app()

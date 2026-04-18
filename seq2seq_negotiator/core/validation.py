
from __future__ import annotations

import math
from dataclasses import dataclass
from transformers import TrainerCallback

from .ddp import is_main_process
from .metrics import compute_binary_f1, parse_issue_vocab_from_source, parse_offer_body, safe_div

@dataclass
class DecodedValidationConfig:
    steps: int = 0
    max_rows: int = 2048
    batch_size: int = 64
    num_beams: int = 1
    max_source_length: int = 512
    max_new_tokens: int = 64

def run_decoded_validation_subset(*, predictor, valid_dataset, config: DecodedValidationConfig) -> dict[str, float]:
    n_rows = min(config.max_rows, len(valid_dataset))
    if n_rows == 0:
        return {
            "decoded_valid_n_rows": 0,
            "decoded_valid_action_accuracy": math.nan,
            "decoded_valid_accept_f1": math.nan,
            "decoded_valid_offer_bid_exact_match": math.nan,
            "decoded_valid_offer_issue_accuracy": math.nan,
        }

    eval_ds = valid_dataset.select(range(n_rows))
    source_texts = eval_ds["source_text"]
    gold_actions = [str(x).upper() for x in eval_ds["action_label"]]
    gold_offer_targets = eval_ds["offer_target_text"]
    pred_actions_all = []
    offer_exact_scores = []
    offer_issue_scores = []

    for start in range(0, n_rows, config.batch_size):
        batch_source_texts = source_texts[start:start + config.batch_size]
        batch_gold_actions = gold_actions[start:start + config.batch_size]
        batch_gold_offer_targets = gold_offer_targets[start:start + config.batch_size]
        batch_pred_actions, batch_pred_offer_bodies = predictor.predict_batch(
            batch_source_texts,
            max_source_length=config.max_source_length,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
        )
        for source_text, gold_action, pred_action, gold_offer_body, pred_offer_body in zip(
            batch_source_texts, batch_gold_actions, batch_pred_actions, batch_gold_offer_targets, batch_pred_offer_bodies
        ):
            pred_actions_all.append(pred_action)
            if gold_action == "OFFER":
                issue_vocab = parse_issue_vocab_from_source(source_text)
                gold_bid = parse_offer_body(str(gold_offer_body or ""), issue_vocab)
                if pred_action != "OFFER":
                    offer_exact_scores.append(0.0)
                    offer_issue_scores.append(0.0)
                else:
                    pred_bid = parse_offer_body(pred_offer_body, issue_vocab)
                    offer_exact_scores.append(float(pred_bid == gold_bid))
                    if gold_bid:
                        per_issue = [float(pred_bid.get(issue) == gold_val) for issue, gold_val in gold_bid.items()]
                        offer_issue_scores.append(sum(per_issue) / len(per_issue))
                    else:
                        offer_issue_scores.append(math.nan)

    n = len(gold_actions)
    action_correct = sum(int(g == p) for g, p in zip(gold_actions, pred_actions_all))
    gold_accept = [g == "ACCEPT" for g in gold_actions]
    pred_accept = [p == "ACCEPT" for p in pred_actions_all]
    tp = sum(int(g and p) for g, p in zip(gold_accept, pred_accept))
    fp = sum(int((not g) and p) for g, p in zip(gold_accept, pred_accept))
    fn = sum(int(g and (not p)) for g, p in zip(gold_accept, pred_accept))
    finite_issue_scores = [x for x in offer_issue_scores if not math.isnan(x)]

    return {
        "decoded_valid_n_rows": int(n),
        "decoded_valid_action_accuracy": safe_div(action_correct, n),
        "decoded_valid_accept_f1": compute_binary_f1(tp, fp, fn),
        "decoded_valid_offer_bid_exact_match": (float(sum(offer_exact_scores) / len(offer_exact_scores)) if offer_exact_scores else math.nan),
        "decoded_valid_offer_issue_accuracy": (float(sum(finite_issue_scores) / len(finite_issue_scores)) if finite_issue_scores else math.nan),
    }

class DecodedValidationCallback(TrainerCallback):
    def __init__(self, *, predictor, valid_dataset, config: DecodedValidationConfig):
        self.predictor = predictor
        self.valid_dataset = valid_dataset
        self.config = config
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or not is_main_process():
            return control
        if self.config.steps <= 0 or state.global_step == 0:
            return control
        if state.global_step % self.config.steps != 0:
            return control
        metrics = run_decoded_validation_subset(
            predictor=self.predictor,
            valid_dataset=self.valid_dataset,
            config=self.config,
        )
        self.trainer.log(metrics)
        return control

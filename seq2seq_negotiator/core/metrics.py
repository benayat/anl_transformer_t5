from __future__ import annotations

import math
import re
from typing import Optional

ISSUE_LINE_RE = re.compile(r"^ISSUE\s+(?P<issue>.+?)\s+VALUES\s+(?P<values>.+)$")
V2_VOCAB_LINE_RE = re.compile(r"^@V\s+(?P<issue>[^:\s]+):(?P<values>.+)$")


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else math.nan


def compute_binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        return math.nan
    return 2.0 * precision * recall / (precision + recall)


def parse_issue_vocab_from_source(source_text: str) -> dict[str, list[str]]:
    """
    Supports both source serializations:
    - v1: ISSUE <name> VALUES a,b,c
    - v2: @V i1:v1|v2|v3
    """
    vocab: dict[str, list[str]] = {}
    for raw_line in str(source_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ISSUE_LINE_RE.match(line)
        if match:
            issue = match.group("issue").strip()
            values = [v.strip() for v in match.group("values").split(",") if v.strip()]
            vocab[issue] = values
            continue
        match = V2_VOCAB_LINE_RE.match(line)
        if match:
            issue = match.group("issue").strip()
            values = [v.strip() for v in match.group("values").split("|") if v.strip()]
            vocab[issue] = values
    return vocab


def canonical_issue_names(issue_vocab: dict[str, list[str]]) -> list[str]:
    return sorted(issue_vocab.keys())


def parse_offer_body(payload: str, issue_vocab: dict[str, list[str]]) -> dict[str, str]:
    """
    Robust parser for both formats:

    v1 multi-line / issue-assignment offers:
      i1 = v3\ni2 = v5
      i1 = v3 i2 = v5 i3 = v1

    v2 compact ordered bids:
      v3,v5,v1

    The return value always maps issue name -> value token.
    """
    payload = (payload or "").strip()
    if not payload:
        return {}

    # Normalize away an optional leading O / ACTION OFFER wrapper.
    upper = payload.upper()
    if upper.startswith("ACTION OFFER"):
        payload = payload[len("ACTION OFFER") :].strip()
    elif upper == "O":
        payload = ""
    elif upper.startswith("O "):
        payload = payload[2:].strip()

    if not payload:
        return {}

    issue_names = canonical_issue_names(issue_vocab)
    if not issue_names:
        return {}

    # v2 compact ordered bid: comma-separated value ids with no explicit issue assignments.
    if "=" not in payload:
        values = [v.strip() for v in payload.split(",") if v.strip()]
        if values and len(values) <= len(issue_names):
            return {issue: value for issue, value in zip(issue_names, values)}

    # v1 line-based parser.
    bid: dict[str, str] = {}
    for line in payload.splitlines():
        line = line.strip().strip("|")
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in issue_vocab and value:
            bid[key] = value
    if bid:
        return bid

    # v1 one-line fallback using known issue names.
    pattern = re.compile(
        r"(" + "|".join(re.escape(x) for x in issue_names) + r")\s*=",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(payload))
    parsed: dict[str, str] = {}
    for i, match in enumerate(matches):
        issue = match.group(1)
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(payload)
        value = payload[value_start:value_end].strip().strip("|")
        if not value:
            continue
        canonical = next((x for x in issue_vocab if x.lower() == issue.lower()), issue)
        parsed[canonical] = value
    return parsed


def parse_single_target_action_text(text: str) -> tuple[str, str]:
    """
    Returns (action, offer_body).
    Supports both v1 and v2 single-target outputs.
    """
    t = (text or "").strip()
    u = t.upper()
    if u in {"A", "ACTION ACCEPT"} or u.startswith("ACTION ACCEPT"):
        return "ACCEPT", ""
    if u == "O":
        return "OFFER", ""
    if u.startswith("ACTION OFFER"):
        return "OFFER", t[len("ACTION OFFER") :].strip()
    if u.startswith("O "):
        return "OFFER", t[2:].strip()
    # fallback: treat as offer body to preserve old behavior
    return "OFFER", t


def serialize_canonical_compact_bid(bid: Optional[dict[str, str]], issue_vocab: dict[str, list[str]]) -> Optional[str]:
    if not bid:
        return None
    issue_names = canonical_issue_names(issue_vocab)
    if not issue_names:
        return None
    try:
        return ",".join(bid[name] for name in issue_names)
    except KeyError:
        return None

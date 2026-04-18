
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = []
    for child in output_dir.iterdir():
        if child.is_dir():
            m = _CHECKPOINT_RE.match(child.name)
            if m:
                candidates.append((int(m.group(1)), child))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def resolve_resume_checkpoint(*, output_dir: Path, resume: str, resume_from_checkpoint: Optional[Path]) -> Optional[str]:
    if resume_from_checkpoint is not None:
        return str(resume_from_checkpoint)
    if resume == "never":
        return None
    if resume == "auto":
        latest = find_latest_checkpoint(output_dir)
        return str(latest) if latest else None
    raise ValueError(f"Unsupported resume mode: {resume}")


from __future__ import annotations

from pathlib import Path
import hashlib
import json
from filelock import FileLock
from datasets import DatasetDict, load_from_disk

def build_token_cache_key(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]

def prepare_or_load_tokenized_dataset(*, cache_root: Path, cache_key_payload: dict, build_fn) -> DatasetDict:
    cache_root.mkdir(parents=True, exist_ok=True)
    key = build_token_cache_key(cache_key_payload)
    target_dir = cache_root / key
    lock_path = cache_root / f"{key}.lock"
    with FileLock(str(lock_path)):
        if target_dir.exists():
            return load_from_disk(str(target_dir))
        ds = build_fn()
        ds.save_to_disk(str(target_dir))
        return ds

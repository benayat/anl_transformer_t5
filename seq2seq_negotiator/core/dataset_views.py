
from __future__ import annotations

from pathlib import Path
from typing import Sequence
from datasets import Dataset, DatasetDict, load_dataset

from .config import resolve_dataset_dir

def _load_single_parquet(path: Path, split_name: str, columns: Sequence[str] | None = None) -> Dataset:
    ds = load_dataset("parquet", data_files={split_name: str(path)}, split=split_name)
    if columns is not None:
        ds = ds.select_columns(list(columns))
    return ds

def load_named_view(*, dataset_dir: str | Path | None, view_name: str, splits: Sequence[str] = ("train", "valid", "test"), columns: Sequence[str] | None = None) -> DatasetDict:
    ds_dir = resolve_dataset_dir(dataset_dir)
    base = ds_dir / view_name
    out = {}
    for split in splits:
        out[split] = _load_single_parquet(base / f"{split}.parquet", split_name=split, columns=columns)
    return DatasetDict(out)

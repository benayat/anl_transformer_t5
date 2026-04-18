
from __future__ import annotations

import os

def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))

def get_rank() -> int:
    return int(os.getenv("RANK", "0"))

def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

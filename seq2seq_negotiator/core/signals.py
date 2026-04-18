
from __future__ import annotations

import signal
import threading
from dataclasses import dataclass

@dataclass
class StopState:
    requested: bool = False
    reason: str | None = None

_STOP_STATE = StopState()
_LOCK = threading.Lock()

def _handle_signal(signum, frame):
    with _LOCK:
        _STOP_STATE.requested = True
        _STOP_STATE.reason = signal.Signals(signum).name

def install_signal_handlers() -> None:
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _handle_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_signal)

def stop_requested() -> bool:
    with _LOCK:
        return _STOP_STATE.requested

def stop_reason() -> str | None:
    with _LOCK:
        return _STOP_STATE.reason

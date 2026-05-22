from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class _Phase:
    label: str
    state: str = "pending"  # pending | active | done


class BackendLoader:
    """Tracks readiness of the search backend while it warms up.

    status: "loading" -> "ready", or "loading" -> "error".
    Thread-safe: the background load thread mutates phase/status under a lock;
    request handlers read a consistent snapshot via ``snapshot()``.
    """

    def __init__(self, phase_labels: list[str]) -> None:
        self._phases = [_Phase(label) for label in phase_labels]
        self.status = "loading"
        self.error: str | None = None
        self._lock = threading.Lock()

    def start_phase(self, index: int) -> None:
        with self._lock:
            self._phases[index].state = "active"

    def finish_phase(self, index: int) -> None:
        with self._lock:
            self._phases[index].state = "done"

    def run(self, load_fn) -> None:
        """Execute load_fn(self); flip to ready on success, error on exception.

        On failure the phase that was active is left as-is (so the loading
        screen shows which step failed) and status/error are set."""
        try:
            load_fn(self)
        except Exception as e:  # noqa: BLE001 — surface any load failure to UI
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
                self.status = "error"
            return
        with self._lock:
            self.status = "ready"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "phases": [{"label": p.label, "state": p.state} for p in self._phases],
            }

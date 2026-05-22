from __future__ import annotations

import pytest

from pixsage.web.loader import BackendLoader


def test_initial_state_is_loading_with_pending_phases():
    loader = BackendLoader(["Loading search model…", "Loading catalog vectors…"])
    snap = loader.snapshot()
    assert snap["status"] == "loading"
    assert snap["error"] is None
    assert [p["label"] for p in snap["phases"]] == [
        "Loading search model…",
        "Loading catalog vectors…",
    ]
    assert all(p["state"] == "pending" for p in snap["phases"])


def test_run_advances_phases_and_becomes_ready():
    loader = BackendLoader(["a", "b"])
    seen = []

    def load_fn(ldr):
        ldr.start_phase(0)
        seen.append(ldr.snapshot()["phases"][0]["state"])  # active
        ldr.finish_phase(0)
        ldr.start_phase(1)
        ldr.finish_phase(1)

    loader.run(load_fn)
    snap = loader.snapshot()
    assert seen == ["active"]
    assert snap["status"] == "ready"
    assert [p["state"] for p in snap["phases"]] == ["done", "done"]


def test_run_records_error_and_leaves_active_phase_visible():
    loader = BackendLoader(["a", "b"])

    def load_fn(ldr):
        ldr.start_phase(0)
        raise RuntimeError("boom")

    loader.run(load_fn)
    snap = loader.snapshot()
    assert snap["status"] == "error"
    assert "boom" in snap["error"]
    assert "RuntimeError" in snap["error"]
    assert snap["phases"][0]["state"] == "active"

from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from pixsage.cli import app


class FakePopen:
    """subprocess.Popen stand-in that exits with `returncode` and writes
    `log_line` to its stdout sink (so _stage_summary can read it)."""

    instances: list["FakePopen"] = []

    def __init__(self, cmd, *, returncode: int = 0, log_line: bytes = b"done. processed=10 skipped=0 errored=0\n",
                 stdout=None, stderr=None, **kwargs):
        self.cmd = cmd
        self.returncode = returncode
        self._terminated = False
        if stdout is not None and hasattr(stdout, "write"):
            stdout.write(log_line)
            stdout.flush()
        FakePopen.instances.append(self)

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        # "still running" for the dashboard so finally-block terminate is exercised
        return None if self._is_dashboard() and not self._terminated else self.returncode

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def _is_dashboard(self) -> bool:
        return any("scripts.dashboard" in part for part in self.cmd)


def _install_fake_popen(monkeypatch, **fake_kwargs):
    FakePopen.instances = []

    def fake(cmd, **kwargs):
        return FakePopen(cmd, **fake_kwargs, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake)


def test_run_invokes_tag_then_embed(tmp_path: Path, monkeypatch) -> None:
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    _install_fake_popen(monkeypatch)

    result = CliRunner().invoke(app, ["run", str(photo_root), "--no-dashboard"])

    assert result.exit_code == 0, result.stdout
    cmds = [p.cmd for p in FakePopen.instances]
    assert len(cmds) == 2
    assert "tag" in cmds[0] and str(photo_root) in cmds[0]
    assert "embed" in cmds[1] and str(photo_root) in cmds[1]


def test_run_creates_logs_under_photoindex(tmp_path: Path, monkeypatch) -> None:
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    _install_fake_popen(monkeypatch)

    result = CliRunner().invoke(app, ["run", str(photo_root), "--no-dashboard"])

    assert result.exit_code == 0, result.stdout
    assert (photo_root / ".photoindex" / "logs" / "tag.log").exists()
    assert (photo_root / ".photoindex" / "logs" / "embed.log").exists()


def test_run_launches_dashboard_by_default(tmp_path: Path, monkeypatch) -> None:
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    _install_fake_popen(monkeypatch)

    result = CliRunner().invoke(app, ["run", str(photo_root)])

    assert result.exit_code == 0, result.stdout
    cmds = [p.cmd for p in FakePopen.instances]
    assert any("scripts.dashboard" in " ".join(c) for c in cmds)
    # Dashboard must be torn down by the finally block.
    dashboard = next(p for p in FakePopen.instances if p._is_dashboard())
    assert dashboard._terminated


def test_run_aborts_on_tag_failure(tmp_path: Path, monkeypatch) -> None:
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    FakePopen.instances = []

    def fake(cmd, **kwargs):
        # Tag fails; embed must not be reached.
        if "tag" in cmd:
            return FakePopen(cmd, returncode=1, **kwargs)
        return FakePopen(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake)

    result = CliRunner().invoke(app, ["run", str(photo_root), "--no-dashboard"])

    assert result.exit_code == 1
    stage_cmds = [p.cmd for p in FakePopen.instances if not p._is_dashboard()]
    assert len(stage_cmds) == 1
    assert "tag" in stage_cmds[0]


def test_run_summary_extracts_done_line(tmp_path: Path, monkeypatch) -> None:
    photo_root = tmp_path / "corpus"
    photo_root.mkdir()
    _install_fake_popen(
        monkeypatch,
        log_line=b"100/100 [eta 00:00]\rdone. processed=42 skipped=3 errored=1\n",
    )

    result = CliRunner().invoke(app, ["run", str(photo_root), "--no-dashboard"])

    assert result.exit_code == 0, result.stdout
    assert "processed=42 skipped=3 errored=1" in result.stdout

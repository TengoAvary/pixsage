from pathlib import Path

from typer.testing import CliRunner

from pixsage.cli import app


def test_stage_launchers_writes_files(tmp_path: Path) -> None:
    folder = tmp_path / "Sony alpha 7c"
    folder.mkdir()
    runner = CliRunner()
    result = runner.invoke(app, ["stage-launchers", str(folder)])
    assert result.exit_code == 0, result.stdout
    assert (folder / "Pixsage Search.bat").exists()
    assert (folder / "Pixsage Search.command").exists()


def test_stage_launchers_rejects_missing_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["stage-launchers", str(tmp_path / "missing")])
    # typer.Argument(... exists=True) makes this exit 2 (usage error)
    assert result.exit_code != 0

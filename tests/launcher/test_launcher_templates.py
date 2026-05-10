from scripts.launcher.launcher_templates import WINDOWS_BAT, MACOS_COMMAND, render


def test_windows_bat_invokes_runtime_pythonw_and_serves_parent_dir() -> None:
    body = render(WINDOWS_BAT, runtime_path=r"%LOCALAPPDATA%\pixsage")
    assert "%LOCALAPPDATA%\\pixsage\\python\\pythonw.exe" in body
    assert "-m pixsage serve" in body
    assert "%~dp0" in body  # parent dir of the .bat
    assert "set PYTHONNOUSERSITE=1" in body  # isolate from host user site-packages


def test_macos_command_invokes_runtime_python_and_serves_parent_dir() -> None:
    body = render(MACOS_COMMAND, runtime_path="$HOME/Library/Application Support/pixsage")
    assert "$HOME/Library/Application Support/pixsage/python/bin/python3" in body
    assert "-m pixsage serve" in body
    assert 'cd "$(dirname "$0")"' in body
    assert "export PYTHONNOUSERSITE=1" in body  # isolate from host user site-packages


def test_render_substitutes_only_known_placeholder() -> None:
    """render() does plain string substitution for {runtime_path}, nothing else."""
    template = "echo {runtime_path} and {other}"
    out = render(template, runtime_path="X")
    assert out == "echo X and {other}"

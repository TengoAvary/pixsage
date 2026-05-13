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


def test_laptop_command_template_invokes_serve_with_no_path() -> None:
    from scripts.launcher.launcher_templates import LAPTOP_MACOS_COMMAND, render
    rendered = render(LAPTOP_MACOS_COMMAND, runtime_path="/Users/test/Library/Application Support/pixsage")
    # No "$PWD" or path argument after `pixsage serve`
    assert "-m pixsage serve" in rendered
    # The last arg on the python invocation should not be a directory path
    line = next(l for l in rendered.splitlines() if "pixsage serve" in l)
    parts = line.split("pixsage serve", 1)[1].strip()
    # Args after serve, if any, should be flags only (start with --) or empty
    if parts:
        assert all(p.startswith("--") for p in parts.split()), f"unexpected args: {parts!r}"


def test_laptop_bat_template_invokes_serve_with_no_path() -> None:
    from scripts.launcher.launcher_templates import LAPTOP_WINDOWS_BAT, render
    rendered = render(LAPTOP_WINDOWS_BAT, runtime_path=r"C:\Users\test\AppData\Local\pixsage")
    assert "-m pixsage serve" in rendered
    line = next(l for l in rendered.splitlines() if "pixsage serve" in l)
    # No quoted path on the line after "pixsage serve"
    after = line.split("pixsage serve", 1)[1].strip()
    if after:
        assert all(p.startswith("--") for p in after.split()), f"unexpected args: {after!r}"

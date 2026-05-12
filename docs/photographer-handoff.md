# Photographer handoff — remaining work

Status as of 2026-05-11. Captures what was validated this session and what's left before the photographer (Intel Mac) can use pixsage.

## Where things stand

**Phase 5 v1 (launcher) is functionally complete.** The full pipeline works:

- `python -m scripts.launcher.install_runtime --target <platform>` produces a portable runtime tree on a target machine (Python 3.12.13 + `pixsage[serve]` + pre-staged HF models) at `%LOCALAPPDATA%\pixsage` (Windows) or `~/Library/Application Support/pixsage` (macOS).
- `pixsage stage-launchers <photo_root>` drops `Pixsage Search.bat` + `Pixsage Search.command` into an indexed folder.
- Double-clicking the launcher invokes `<runtime>/python -m pixsage serve <folder>`. Browser opens to the search webapp.

The gated end-to-end smoke test (`tests/launcher/test_smoke.py` with `PIXSAGE_LAUNCHER_SMOKE=1`) **passed cleanly** at commit `1070185` — 1 passed in 7:24, full chain validated on the Windows dev box.

## What this session found and fixed

The smoke test caught **two missing runtime deps** in the `[serve]` extra that were transparently satisfied on the dev workstation:

| Missing dep | Why it failed | Why dev workstation didn't show it |
|---|---|---|
| `torchvision` | `SiglipImageProcessorFast` imports torchvision at construction time when `AutoProcessor` is built with `use_fast=True`. Server crashed at startup. | `[taggers]` pulls it transitively via `ram` (recognize-anything). |
| `python-multipart` | FastAPI requires it at app-startup time for routes using `Form()` (the `/search` POST does). | Some ambient package on the dev box (likely gradio) had pulled it. |

Both fixed in `pyproject.toml` (commit `1070185`). The photographer's fresh install would have hit these.

## What we explored and rejected

After the smoke test went green, we explored what a more polished photographer-facing UX would look like — broadening scope to include a native `.app`, multi-corpus search with persistent registry, codesigning, etc. Two design rounds in, we cut all of it. Reasons:

- **The .app + Rust + tray icon + cross-compile route** would have been days of work for a single user. Overkill.
- **Multi-corpus / cross-trip search with a persistent registry** is interesting but solves a problem the photographer hasn't asked for. He has one drive.
- **A blocking osascript modal dialog for "quit UI"** is exactly the wrong shape — sits in the foreground over his work.
- **The "quittable process" concern was Windows-specific.** On Mac, `.command` opens a Terminal window for the lifetime of serve, and the X button on that Terminal IS the quit affordance. v1 already does this.

The v1 Mac launcher is sufficient as-is. The remaining gap is not the launcher; it's the runtime build.

## Actual remaining work (in order)

### 1. Intel Mac runtime target  (~half day)

`build_runtime.py` and `install_runtime.py` currently know about `windows-x64` and `macos-arm64`. The photographer is on **Intel Mac**, so we need a `macos-x86_64` target.

- Look up the right `python-build-standalone` release filename pattern for Intel-Mac (`cpython-3.12.13+...-x86_64-apple-darwin-install_only_stripped.tar.gz` is the likely shape; verify against the actual GitHub release tag we're pinned to).
- Add the target to the `--target` choices in `install_runtime.py` and the PBS URL builder in `build_runtime.py`.
- Update `download_models.py` if it gates on platform (it shouldn't, but verify).
- Smoke test (`tests/launcher/test_smoke.py`) currently picks target by `sys.platform`. Update to also branch on `platform.machine()` so Intel-Mac and Apple-Silicon-Mac pick the right PBS tarball.
- pip should resolve compatible torch + torchvision wheels for `darwin-x86_64` without further changes. Both ship Intel-Mac wheels.

This is the hard blocker — the photographer can't install the runtime today without this.

### 2. Plan 2.5: SigLIP2 text-tower extraction  (~3 hours)

The full SigLIP2-so400m model is ~1.8 GB. At serve time we only use the **text tower** (~280 MB) — the image tower is dead weight on the photographer's laptop.

- `pixsage.embedders.siglip2`: when only query-encoding is needed (serve), load `SiglipTextModel.from_pretrained` instead of the full `AutoModel`. Gate behind a constructor flag (`query_only=True` or similar).
- `scripts/launcher/download_models.py`: pre-extract just the text-tower checkpoint into the runtime archive. Optionally save a stripped version on disk so HF doesn't re-download the full model.
- `[serve]` extras stay the same (transformers can load either form).
- Verify smoke test still passes after the change.

Saves ~1.5 GB on every photographer install. Worth doing before handoff regardless of urgency.

### 3. iPhone-corpus path-walker / format quirks (~half day total)

Running tag on `E:\iphone 15 pro` (5635 files, 4656 candidates) produced 129 errored rows in five distinct categories — all real-world iPhone-ecosystem quirks worth handling at the source rather than via post-run `pixsage cleanup`:

| Count | Class | Suggested fix |
|---:|---|---|
| 66 | rawpy `b'Unsupported file format or not RAW file'` on some iPhone ProRAW DNGs | Catch the rawpy error; fall back to PIL+pillow-heif's DNG path if it can read the file's embedded preview. Currently we hard-fail. |
| 36 | exiftool `Not a valid JPG (looks more like a TIFF)` on `.JPG` files that are actually HEIC/TIFF | Detect actual file format via magic bytes before invoking exiftool's `-JPG` mode. Use the right exiftool format flag or skip the XMP write step (catalog still records the tags). |
| 22 | PIL `cannot identify image file` — 6 are macOS AppleDouble `._*` companions, 16 are corrupt/unsupported | Skip `._*` in the path enumerator (same idea as the existing `.DS_Store` skip — one-line fix). Remaining 16 can stay errored. |
| 4 | `Unsupported BMP pixel depth (0)` | Probably also covered by the magic-byte detection above — these are likely misnamed files. |
| 1 | exiftool `Not a valid DNG (looks more like a JPEG)` | Same magic-byte detection. |

The 22-row PIL-cannot-identify class is the cheapest fix (one-line filter for `._*`) and the highest-value because it's a known macOS-on-exFAT artefact that hits every drive transferred from a Mac. Worth doing first.

The HEIC-as-JPG and rawpy-can't-DNG classes are deeper changes (probe file format at read time) but they're the difference between "iPhone corpus indexes cleanly" and "iPhone corpus errors on ~2% of files." Worth doing before the next iPhone corpus.

### 4. Cosmetic polish on the `.command` (~optional, ~30 min)

The Terminal that pops up on launcher click currently scrolls server logs. Could be cleaner — banner that says "Pixsage Search running. Close this window to stop." with logs routed to a file. Not blocking. Skip unless it bothers anyone.

## Handoff sequence (once 1+2 are done)

1. Jack: physical access to photographer's Mac (or remote shell session).
2. Jack: clone pixsage or copy the source tree to the Mac.
3. Jack: `python -m scripts.launcher.install_runtime --target macos-x86_64` on the Mac. ~10 min, ~280 MB download.
4. Jack: on his Windows workstation, `pixsage stage-launchers E:\Sony alpha 7c` to drop `Pixsage Search.command` into the indexed drive.
5. Photographer: plugs drive into Mac, opens the folder, double-clicks `Pixsage Search.command`. Browser opens. Terminal stays open as the "running" indicator. Closes Terminal to stop.

## What's explicitly *not* on the roadmap

(So the next session doesn't re-litigate.)

- Native `.app` bundle, codesigning, notarization. Skip unless the v1 `.command` UX proves a real problem in practice.
- Multi-corpus / cross-trip search. Not asked for.
- Persistent corpus registry, menu-bar item, tray icon.
- Auto-update of the runtime.
- Windows photographer launcher. He's Mac-only.
- Phase 2 (pHash identification). Still deferred.

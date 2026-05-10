# Clickable launcher for pixsage search — design

**Date:** 2026-05-10
**Status:** approved
**Scope:** Phase 5 — packaging polish. Make the search webapp openable by a non-technical user via a double-clickable launcher, on both Windows and Mac, with zero ongoing drive overhead.

## Goal

A photographer client receives an external drive containing her processed photos (raws + sidecar XMPs + a `.photoindex/` catalog produced by `pixsage tag` + `pixsage embed` on the workstation). She should be able to:

1. Plug the drive into her own machine (Windows or Mac).
2. Open any indexed folder.
3. Double-click a single file (`Pixsage Search.exe` / `Pixsage Search.app`).
4. See the existing pixsage search webapp in her default browser, scoped to that folder's catalog.

No terminal use, no Python installation step she has to perform deliberately, no path picker, no "which catalog?" dialog. The first launch ever triggers a one-time runtime install to her local machine; every subsequent launch is instant.

## Non-goals

- Running the tag / embed / geolocate pipeline on her machine. Processing stays on the workstation; she only runs `serve`.
- Letting her edit the catalog, write XMPs, or modify state. Read-only view.
- Replacing or duplicating Lightroom features. Search-and-discovery only.
- Supporting Linux as a target. Windows + Mac only.
- Supporting Intel Macs in v1. Apple Silicon (ARM64) only.
- Notarization / paid code-signing certs. First-launch warnings are acceptable.

## User experience

### Day-one handoff

Jack hands the photographer the external drive. The drive's root looks like:

```
PIXSAGE/                       (exFAT, cross-OS readable/writable)
├── Sony alpha 7c/
│   ├── .photoindex/           catalog + vectors (~10 MB)
│   ├── Pixsage Search.exe     ~1 MB Windows launcher
│   ├── Pixsage Search.app/    ~500 KB Mac launcher
│   ├── DSC_0001.ARW
│   ├── DSC_0001.xmp
│   └── ...
└── (other indexed folders, same shape)
```

Each indexed folder has *both* launchers — Windows ignores the `.app`, Mac ignores the `.exe`. Total per-folder overhead: ~1.5 MB.

### First launch ever (one time per machine)

1. Photographer opens an indexed folder, double-clicks `Pixsage Search`.
2. On Windows: SmartScreen warning ("Unrecognized app"). She clicks "More info → Run anyway" once. On Mac: Gatekeeper warning ("unidentified developer"). She right-clicks → Open the first time. Each OS remembers per binary.
3. Launcher detects no runtime installed at the standard local-runtime path. Opens a small native dialog:
   > **First-time setup**
   > Pixsage needs to install a runtime (~2.5 GB) on your computer.
   > This takes about 5 minutes on a normal internet connection.
   > [Install] [Cancel]
4. She clicks Install. Progress bar shows download from a public URL (GitHub Release / R2 / S3). Runtime + models unpacked into the local-runtime path (see below).
5. When done, launcher proceeds normally: spawns Python, waits for FastAPI to bind, opens default browser to `http://127.0.0.1:8765/`.

### Daily launches

1. Plug in drive.
2. Open an indexed folder, double-click `Pixsage Search`.
3. ~3 seconds later: browser opens to the search webapp scoped to that folder's catalog. A small tray/menubar icon ("Pixsage: Sony alpha 7c") appears with a Quit option.
4. Close the browser tab when done; server stays running until she Quits from the tray. Re-clicking the launcher within a session just re-opens the browser tab.

### Multiple indexed folders

Each folder is its own launcher. Two folders, two `Pixsage Search` files, two independent searches if she opens both. No global picker, no shared "which catalog?" state. Two tray icons (one per server). Default port is 8765; collisions auto-increment.

## Architecture

### Components

#### 1. Local runtime directory

Installed once per user machine. Contains everything needed to run `pixsage serve` without an OS-installed Python.

- **Windows:** `%LOCALAPPDATA%\pixsage\` (resolves to `C:\Users\<user>\AppData\Local\pixsage\`)
- **Mac:** `~/Library/Application Support/pixsage/`

Layout:
```
pixsage/                         (local runtime root)
├── python/                      python-build-standalone tarball, unpacked
│   ├── bin/  (Mac) | python.exe (Windows)
│   └── lib/
├── site-packages/               pixsage + deps installed via pip --target
├── models/                      SigLIP2 + MiniLM weights (HF cache layout)
└── version.txt                  semver of installed pixsage runtime
```

Size: ~780 MB after Plan 2 (Python+CPU-torch+transformers+deps ~500 MB, models ~280 MB, pixsage source ~1 MB).

**Sizing note:** an earlier draft of this spec assumed ~1.8 GB of models — that was the full SigLIP2-so400m + MiniLM. Serve only ever calls `model.get_text_features(...)` (text tower) — the vision tower is unused at serve time. A post-Plan-2 follow-up extracts the SigLIP2 text tower as a standalone checkpoint (~200 MB instead of ~1.7 GB), bringing models to ~280 MB. Until that follow-up ships, Plan 2's `download_models.py` pulls the full SigLIP2 (~1.8 GB models, ~2.3 GB total runtime).

#### 2. Per-folder launcher binary

A native single-file executable that lives in each indexed folder next to `.photoindex/`. Two builds:

- **Windows:** `Pixsage Search.exe`, ~1 MB. Built in Rust (or Go).
- **Mac:** `Pixsage Search.app` bundle, ~500 KB. Built in Rust or Swift, ad-hoc signed.

Both implement the same logic:

1. Determine `photo_root` = directory containing the launcher binary itself.
2. Determine local runtime path (OS-specific constant).
3. If runtime missing or version below required: trigger first-time-setup flow (download + unpack).
4. Set env: `HF_HOME=<runtime>/models`, `HF_HUB_OFFLINE=1`, `PYTHONPATH=<runtime>/site-packages`.
5. Spawn: `<runtime>/python/python -m pixsage serve --photo-root <photo_root> --no-open --port <auto>`.
6. Poll `127.0.0.1:<port>` until ready.
7. Open default browser at the ready URL.
8. Show tray/menubar icon with name "Pixsage: <folder basename>" and a Quit action that stops the spawned process.

#### 3. Runtime installer

Triggered by the launcher on first run. Logic:

1. Detect OS + arch (Windows x64, Mac arm64).
2. Download a single runtime tarball from a fixed URL (e.g. `https://github.com/<repo>/releases/download/v<X>/pixsage-runtime-<os>-<arch>.tar.zst`).
3. Verify size + sha256 against an embedded manifest.
4. Unpack into the local-runtime path atomically (write to `.tmp/`, rename).
5. Write `version.txt`.

Resumable downloads via HTTP Range requests; corrupted install detected by sha256 mismatch and offered a Repair option.

#### 4. Path-translation layer in `pixsage serve`

The catalog stores absolute paths as recorded at embed time on the workstation (e.g. `E:\Sony alpha 7c\DSC_1234.ARW`). On the photographer's machine the same file lives at a different absolute path (`F:\Sony alpha 7c\DSC_1234.ARW` on Windows with a different drive letter, or `/Volumes/Sony alpha 7c/Sony alpha 7c/DSC_1234.ARW` on Mac).

Solution: catalog grows a `meta` table storing the `photo_root` used at embed time. At serve startup, the launcher passes `--photo-root` (its own parent dir) and the web app computes a prefix substitution: `<stored-photo-root> → <runtime-photo-root>`. Apply at every catalog read that surfaces a path to the browser (thumbnails, photo detail, "more like this" file resolution).

If the substitution doesn't produce a file that exists, fall through to trying the catalog's stored path verbatim (handles edge cases where the photo was never moved). Surface an in-app warning if neither resolves.

### Build pipeline (workstation-side)

Five scripts, all under `scripts/launcher/`:

1. **`build_runtime.py --target {windows-x64,macos-arm64} --out <dir>`** *(shipped in Plan 2)*
   - Uses [python-build-standalone](https://github.com/astral-sh/python-build-standalone) tarballs (downloadable, prebuilt portable Python for both targets).
   - Runs `pip install --target <dir>/site-packages` against `pyproject.toml`'s `[serve]` extras (added in Plan 2 — slimmer than `[taggers]`+`[search]` since rawpy + ram are tag/embed-time only).

2. **`download_models.py --out <dir>`**
   - Pre-downloads SigLIP2 + MiniLM weights into a clean HF cache layout in `<dir>/models/`.

3. **`build_launcher.py --target {windows,macos-arm64}`**
   - Compiles the Rust launcher binary. Embeds the runtime version + URL + sha256 manifest at compile time.
   - On Mac: assembles `.app` bundle with Info.plist, ad-hoc codesigns with `codesign --sign -`.

4. **`stage_folder.py <photo-root>`**
   - Drops `Pixsage Search.exe` + `Pixsage Search.app/` next to the folder's `.photoindex/`.
   - Idempotent — safe to re-run after a launcher version bump.
   - Run automatically at the end of `pixsage embed` (or as a separate verb `pixsage stage-launcher <root>`).

5. **`publish_runtime.py --version <X>`**
   - Assembles the runtime tarballs (one per OS/arch), uploads to GitHub Release or R2.
   - Generates the manifest (sizes + sha256s) consumed by the launcher binaries.

### Data flow at runtime

```
double-click Pixsage Search.exe
  └─ launcher: resolve photo_root = its parent dir
  └─ launcher: check %LOCALAPPDATA%\pixsage\version.txt
     ├─ missing or stale → run installer (download, unpack, write version.txt)
     └─ present and current → proceed
  └─ launcher: spawn python.exe -m pixsage serve --photo-root <photo_root> --no-open --port <free>
  └─ pixsage.web.app.build_app() reads meta.photo_root_at_embed from catalog
  └─ pixsage.web.app installs path-substitution middleware
  └─ uvicorn binds to 127.0.0.1:<port>
  └─ launcher polls, opens browser when /healthz returns 200
  └─ launcher attaches tray icon; waits for Quit
```

## Testing strategy

- **Unit (existing test suite):** path-substitution layer needs unit tests against fake catalogs whose `meta.photo_root_at_embed` differs from the runtime `photo_root`. Cover: Windows drive-letter swap (E: → F:), Windows → Mac mount-point translation, missing-file fallback to stored path.
- **Integration:** an end-to-end test that runs the launcher binary against a fixture drive (a temp dir staged to look like the photographer's drive). Verifies: launcher spawns Python, server binds, `/healthz` reachable. Marked as gated on whether the launcher binary has been built for the test machine's OS.
- **Manual smoke on each handoff:** stage a test drive, plug into a clean Windows VM + a clean Mac VM, first-launch flow end-to-end. Documented checklist in `docs/launcher-smoke-test.md`.

## Open decisions deferred to implementation

- Tray icon library choice (per OS). For Rust: `tray-icon` or `tao` ecosystem. For Mac alone, `cocoa` bindings. Pick at implementation time.
- Whether `pixsage stage-launcher` is its own CLI verb or runs implicitly at the tail end of `pixsage embed`. Probably both — implicit by default, explicit verb for re-staging after a launcher version bump.
- Concrete download host. GitHub Release is the obvious zero-cost default; revisit if outbound bandwidth becomes a concern.
- Port-collision handling: auto-increment from 8765, or surface a config knob. Auto-increment is simpler.

## Risks

- **`python-build-standalone` size.** The portable Python is ~50 MB but with `pip install --target` of torch + transformers it balloons. CPU-only torch wheel is ~200 MB; transformers + tokenizers + sentence-transformers another ~100 MB. Worst case the runtime tarball lands at ~700 MB compressed. Acceptable but worth measuring early.
- **Mac Gatekeeper friction.** Ad-hoc signing means the photographer sees one warning per launcher binary. If she has many indexed folders this gets annoying. Mitigation if it bites: paid Apple Developer + notarize.
- **Windows SmartScreen reputation.** Unsigned + low download volume triggers warnings. Mitigation: code-signing cert ($70/yr from Sectigo) eliminates this. Defer until first complaint.
- **Drive removed mid-session.** Server keeps running, thumbnails 404, photo detail 404. Need a clear in-UI banner: "drive disconnected — reconnect to continue." Cheap to add.
- **Future runtime updates.** Each launcher binary embeds the required runtime version. Bumping the required version means re-staging every indexed folder with a new launcher binary, which is one `pixsage stage-launcher <root>` per folder. Acceptable.

## Out of scope (future)

- Air-gapped install (no internet on the photographer's machine). The launcher could be extended with a `--runtime <path>` flag accepting a drive-resident `_PixsageRuntime/` directory; flip to that mode only if needed.
- Migrating an already-installed local runtime to a different location.
- Auto-update of the local runtime when a new version is published. v1 prompts on next launch only when the launcher's embedded version pin advances past `version.txt`.
- A "process new photos" button in the UI. Pipeline stays on the workstation.

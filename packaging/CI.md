# CI / CD

## Rust CI

`.github/workflows/ci.yml` runs on pull requests, main pushes and manual dispatch. The reusable `engine.yml` installs Rust 1.97.1, runs formatting, Clippy with warnings denied, unit/doc tests, locked release builds and executable smoke tests on Windows x64, Ubuntu 22.04 x64 and macOS ARM64. CI does not install training dependencies.

## Download bundles

Run **Download bundles** manually to build candidate archives. Download them from the workflow artifacts (retained for 14 days); this does not create a Release.

For a release:

1. Update `Cargo.toml` package version and refresh/commit `Cargo.lock`; keep Python package metadata and `training/uv.lock` consistent if their version is changed too.
2. Merge and confirm Rust CI is green.
3. Push a matching tag, e.g. `v0.1.0` for Cargo version `0.1.0`.
4. The release workflow re-runs all Rust gates, builds the engine once and packages both Engine and Training from that same checkout, and tests the **extracted** bundle using uv and its real launcher. The test checks the GUI-selected engine path, including extraction paths with spaces.
5. Only after all three platforms pass, a draft GitHub Release is created with six archives (two per platform) and six SHA-256 files. Review and manually publish it.

A tag/version mismatch blocks the build. Existing releases are not overwritten automatically; a failed publication rerun needs review rather than silently replacing assets. No release is created on PRs or branch pushes. The jobs only request repository read access, except the isolated release job with `contents: write`. Enable Actions in the repository; optionally require Rust CI checks in branch rules. No personal token is needed.

The release smoke test installs locked CPU dependencies, so allow download time and storage. It does not certify CUDA/MPS hardware or render a real voicebank. Proprietary voicebanks/dictionaries must not be added to CI. Current archives are unsigned and not notarized. The Linux baseline is Ubuntu 22.04; older libc targets require a separate compatibility build. Intel macOS is excluded because the pinned PyTorch package has no matching wheel.

## Local packaging

After building a native engine, run (Python 3.11+):

```text
python packaging/build_bundle.py --engine target/release/opensv --target aarch64-apple-darwin
```

Windows users can run `packaging/package.ps1`, which builds a locked x64 engine and uses uv to run the same packager. Local archives go under `local/packages/` and never enter Git. Existing archive names are protected against replacement. `BUILD.json` records HEAD; build local release candidates only from a clean checkout. CI uses a clean checkout automatically.

## Two download types

- `OpenSV-engine-VERSION-PLATFORM`: standalone engine, CLI documentation, license, build metadata and corresponding Rust source. No Python training runtime or GUI launcher.
- `OpenSV-training-VERSION-PLATFORM`: the same engine plus Python training source, dependency lock files and uv launchers. This is the recommended download for training/GUI users; no second download is needed.

Windows uses `.zip`; Linux and macOS use `.tar.gz`. Each archive has its own `.sha256`. Both kinds share the engine hash, version and commit. `build_bundle.py` defaults to `--kind all`; `--kind engine` or `--kind training` builds only one type. Release validation requires Rust and Python package versions to match. Engine smoke tests do not install Python dependencies; Training smoke tests additionally install the locked environment and start the GUI.

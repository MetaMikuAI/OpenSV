"""Build a platform-specific source + engine bundle using only the standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile

TARGETS = {
    "x86_64-pc-windows-msvc": "windows-x64",
    "x86_64-unknown-linux-gnu": "linux-x64",
    "aarch64-apple-darwin": "macos-arm64",
}
REPO = Path(__file__).resolve().parents[1]


def build(engine, target, output, revision, kind="training"):
    if kind not in ("engine", "training"):
        raise ValueError("Unknown bundle kind")
    version = tomllib.loads((REPO / "Cargo.toml").read_text())["package"]["version"]
    if not re.fullmatch(r"[0-9A-Za-z.+-]+", version):
        raise ValueError("Unsafe package version")
    engine = engine.resolve(strict=True)
    executable = "opensv.exe" if target.endswith("windows-msvc") else "opensv"
    observed = subprocess.check_output([str(engine), "--version"], text=True).strip()
    if observed != f"opensv {version}":
        raise ValueError(f"Engine/source version mismatch: {observed}")
    output.mkdir(parents=True, exist_ok=True)
    name = f"OpenSV-{kind}-{version}-{TARGETS[target]}"
    suffix = ".zip" if executable.endswith(".exe") else ".tar.gz"
    archive = output / (name + suffix)
    if archive.exists():
        raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(prefix="opensv-bundle-") as temporary:
        root = Path(temporary) / name
        root.mkdir()
        shutil.copy2(engine, root / executable)
        (root / executable).chmod(0o755)
        for name in ("LICENSE", "README.en.md"):
            shutil.copy2(REPO / name, root / name)
        shutil.copy2(REPO / "packaging" / ("ENGINE.md" if kind == "engine" else "README.md"), root / "README.md")
        if kind == "training":
            launchers = ("start.cmd", "start-cuda.cmd") if executable.endswith(".exe") else ("start.sh",)
            if target.endswith("apple-darwin"):
                launchers += ("start.command",)
            for launcher in launchers:
                shutil.copy2(REPO / "packaging" / launcher, root / launcher)
                (root / launcher).chmod(0o755)
            training = root / "training"
            training.mkdir()
            for name in ("pyproject.toml", "uv.lock", ".python-version", "README.md"):
                shutil.copy2(REPO / "training" / name, training / name)
            package = REPO / "training/opensv_train"
            for source in sorted(package.rglob("*")):
                if source.is_file() and source.suffix in (".py", ".html") and "__pycache__" not in source.parts:
                    destination = training / "opensv_train" / source.relative_to(package)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        # Include the corresponding Rust source and build files for the GPL engine.
        source_root = root / "engine-source"
        source_root.mkdir()
        for name in ("Cargo.toml", "Cargo.lock", "LICENSE"):
            shutil.copy2(REPO / name, source_root / name)
        shutil.copytree(REPO / "src", source_root / "src")
        shutil.copytree(REPO / "packaging", source_root / "packaging",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        metadata = dict(kind=kind, version=version, target=target, revision=revision,
                        engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest())
        (root / "BUILD.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if suffix == ".zip":
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for item in sorted(root.rglob("*")):
                    if item.is_file():
                        bundle.write(item, item.relative_to(root.parent))
        else:
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(root, arcname=root.name)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "local/packages")
    parser.add_argument("--kind", choices=("engine", "training", "all"), default="all")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    kinds = ("engine", "training") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        print(build(args.engine, args.target, args.output, revision, kind))

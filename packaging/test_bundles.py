"""Offline packaging tests: fake engine bytes, no Rust compilation or downloads."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

import build_bundle


class BundleTests(unittest.TestCase):
    def test_all_platforms_have_distinct_engine_and_complete_training_packages(self):
        self._assert_all_platform_bundles()

    def test_archive_permissions_do_not_depend_on_host(self):
        original = tarfile.TarFile.gettarinfo

        def without_posix_permissions(bundle, *args, **kwargs):
            member = original(bundle, *args, **kwargs)
            if member is not None:
                member.mode = 0o777 if member.isdir() else 0o666
            return member

        with patch.object(tarfile.TarFile, "gettarinfo", without_posix_permissions):
            self._assert_all_platform_bundles()

    def _assert_all_platform_bundles(self):
        version = tomllib.loads((build_bundle.REPO / "Cargo.toml").read_text())["package"]["version"]
        for target in build_bundle.TARGETS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                engine = base / "fake-engine"
                engine.write_bytes(b"packaging fixture, not an executable")
                hashes = []
                for kind in ("engine", "training"):
                    with patch.object(build_bundle.subprocess, "check_output", return_value=f"opensv {version}\n"):
                        archive = build_bundle.build(engine, target, base / "dist", "test-revision", kind)
                    self.assertIn(f"OpenSV-{kind}-{version}-", archive.name)
                    if archive.suffix == ".zip":
                        with zipfile.ZipFile(archive) as bundle:
                            contents = {n.split("/", 1)[1]: bundle.read(n) for n in bundle.namelist()}
                    else:
                        with tarfile.open(archive) as bundle:
                            contents = {n.name.split("/", 1)[1]: bundle.extractfile(n).read()
                                        for n in bundle.getmembers() if n.isfile()}
                            executable = next(n for n in bundle.getmembers() if n.name.endswith("/opensv"))
                            self.assertEqual(executable.mode, 0o755)
                            members = {n.name.split("/", 1)[1]: n for n in bundle.getmembers()
                                       if "/" in n.name}
                            for member in bundle.getmembers():
                                if member.isdir():
                                    self.assertEqual(member.mode, 0o755)
                            self.assertEqual(members["README.md"].mode, 0o644)
                            if kind == "training":
                                self.assertEqual(members["start.sh"].mode, 0o755)
                                if target.endswith("apple-darwin"):
                                    self.assertEqual(members["start.command"].mode, 0o755)
                    metadata = json.loads(contents["BUILD.json"])
                    self.assertEqual(metadata["kind"], kind)
                    self.assertEqual(metadata["target"], target)
                    hashes.append(metadata["engine_sha256"])
                    binary = "opensv.exe" if "windows" in target else "opensv"
                    self.assertEqual(contents[binary], engine.read_bytes())
                    self.assertIn("LICENSE", contents)
                    self.assertIn("engine-source/Cargo.toml", contents)
                    if kind == "training":
                        for name in ("training/pyproject.toml", "training/uv.lock", "training/.python-version",
                                     "training/opensv_train/gui.py", "training/opensv_train/web/index.html"):
                            self.assertIn(name, contents)
                        launcher = "start.cmd" if "windows" in target else "start.sh"
                        self.assertIn(b"--engine", contents[launcher])
                    else:
                        self.assertFalse(any(n.startswith("training/") for n in contents))
                        self.assertFalse(any(n.startswith("start") for n in contents))
                    # Text reads normalize CRLF and would hide Windows-only defects.
                    checksum_bytes = archive.with_name(archive.name + ".sha256").read_bytes()
                    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                    self.assertEqual(checksum_bytes, f"{digest}  {archive.name}\n".encode("utf-8"))
                self.assertEqual(hashes[0], hashes[1])
                self.assertEqual(len(list((base / "dist").iterdir())), 4)

    def test_mismatched_engine_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "fake-engine"
            engine.touch()
            with patch.object(build_bundle.subprocess, "check_output", return_value="opensv 999.0.0"):
                with self.assertRaisesRegex(ValueError, "version mismatch"):
                    build_bundle.build(engine, "aarch64-apple-darwin", root / "dist", "test", "engine")
            self.assertFalse((root / "dist").exists())

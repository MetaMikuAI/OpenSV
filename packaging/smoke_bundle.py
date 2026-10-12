"""Verify the extracted download, its launcher, bundled engine and local GUI."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request


def smoke(root):
    root = root.resolve()
    info = json.loads((root / "BUILD.json").read_text())
    windows = info["target"].endswith("windows-msvc")
    engine = root / ("opensv.exe" if windows else "opensv")
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == info["engine_sha256"]
    for args in (["--version"], ["--help"], ["train", "--help"]):
        subprocess.run([str(engine), *args], check=True, timeout=30)
    if info["kind"] == "engine":
        assert not (root / "training").exists(), "Engine bundle must not include training runtime"
        assert not any(root.glob("start*")), "Engine bundle must not include GUI launchers"
        print("Engine bundle smoke test passed:", info["target"])
        return
    assert info["kind"] == "training", info
    for required in ("training/pyproject.toml", "training/uv.lock", "training/.python-version",
                     "training/opensv_train/gui.py", "training/opensv_train/web/index.html"):
        assert (root / required).is_file(), required
    subprocess.run(["uv", "sync", "--project", str(root / "training"), "--locked"], check=True)
    launcher = (["cmd", "/c", str(root / "start.cmd")] if windows
                else [str(root / "start.sh")])
    # Launch outside the bundle to catch accidental reliance on cwd or PATH.
    with tempfile.TemporaryDirectory() as directory:
        logfile = Path(directory) / "gui.log"
        with logfile.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([*launcher, "--no-browser"], cwd=directory,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    text = logfile.read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"http://127\.0\.0\.1:\d+/#([^\s]+)", text)
                    if match:
                        base, token = match.group().split("/#", 1)
                        request = urllib.request.Request(base + "/api/defaults",
                                                         headers={"Origin": base, "X-OpenSV-Token": token})
                        with urllib.request.urlopen(request, timeout=10) as response:
                            defaults = json.load(response)
                        assert Path(defaults["engine"]).resolve() == engine.resolve(), defaults
                        print("Bundle smoke test passed:", info["target"])
                        return
                    if process.poll() is not None:
                        raise RuntimeError(text)
                    time.sleep(.2)
                raise TimeoutError(logfile.read_text(encoding="utf-8", errors="replace"))
            finally:
                if windows and process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
                elif process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    smoke(parser.parse_args().root)

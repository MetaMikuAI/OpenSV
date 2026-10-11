"""校验 Rust 声学输出并打包训练数据。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import digest
from .format import Model
from .network import AcousticCore


def pack(output):
    torch.set_num_threads(4)
    model = output / "base_model.dnni"
    core = AcousticCore(Model.read(model).root)
    samples, signatures = [], set()
    maximum_error = 0.0
    supervision = None
    for path in sorted(output.glob("*/sample.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent
        mode = row.get("supervision", "features")
        if mode not in ("audio", "features") or supervision not in (None, mode):
            raise ValueError("Mixed or unsupported supervision")
        supervision = mode
        counts = np.asarray(row["counts"])
        languages = np.asarray(row["languages"])
        frames = row["frames"]
        if row["version"] != 1 or row["phone_channels"] != 108 or row["split"] not in ("train", "validation"):
            raise ValueError("Unsupported capture metadata")
        if (not isinstance(frames, int) or not 0 < frames <= 32768
                or counts.ndim != 1 or not len(counts) or counts.dtype.kind not in "iu"
                or languages.dtype.kind not in "iu" or (counts <= 0).any()
                or counts.sum() != frames or counts.shape != languages.shape):
            raise ValueError("Invalid frame counts")
        counts, languages = counts.astype(np.int64), languages.astype(np.int64)
        arrays = {"counts": counts, "languages": languages}
        shapes = [("phones", (len(counts), 108)), ("pitch", (frames, 1)),
                  ("noise", (frames, 4)), ("baseline", (frames, 71))]
        if mode == "features":
            shapes.append(("target", (frames, 71)))
        else:
            arrays["log_f0"] = np.asarray(row["log_f0"], dtype=np.float32).reshape(frames, 1)
        for name, shape in shapes:
            value = np.fromfile(directory / f"{name}.f32", dtype="<f4").reshape(shape)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name}")
            arrays[name] = value
        signature = hashlib.sha256(b"".join(arrays[key].tobytes() for key in
                                   ("phones", "counts", "languages", "pitch"))).hexdigest()
        if signature in signatures:
            raise ValueError("Repeated phrase inputs; separate training and validation material")
        signatures.add(signature)
        with torch.no_grad():
            context = core.frame_context(torch.from_numpy(arrays["phones"]), counts.tolist(),
                                         languages.tolist(), torch.from_numpy(arrays["pitch"]))
            prediction = core(context, torch.from_numpy(arrays["noise"])).numpy()
        error = float(np.max(np.abs(prediction - arrays["baseline"])))
        maximum_error = max(maximum_error, error)
        if not np.allclose(prediction, arrays["baseline"], rtol=2e-4, atol=2e-4):
            raise ValueError(f"Rust/PyTorch acoustic mismatch: {error}")
        destination = output / f"{directory.name}.npz"
        with destination.open("xb") as stream:
            np.savez(stream, **arrays)
        sample = {"id": directory.name, "split": row["split"], "features": destination.name}
        if mode == "audio":
            sample["audio"] = (directory / "target.wav").relative_to(output).as_posix()
        samples.append(sample)
    if {row["split"] for row in samples} != {"train", "validation"}:
        raise ValueError("Both training and validation samples are required")
    manifest = {"version": 1, "model_sha256": digest(model), "samples": samples, "supervision": supervision,
                "capture": {"engine": "opensv", "pitch_mode": "notes", "timing": "student",
                            "rust_torch_max_error": maximum_error}}
    with (output / "dataset.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest["capture"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    pack(parser.parse_args().output)


if __name__ == "__main__":
    main()

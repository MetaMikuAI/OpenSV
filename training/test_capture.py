import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from opensv_train.capture import pack


class CoreStub:
    def frame_context(self, phones, counts, languages, pitch):
        return pitch

    def __call__(self, context, noise):
        return torch.zeros((len(context), 71))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "base_model.dnni").write_bytes(b"test-model")
        self.model = patch("opensv_train.capture.Model.read").start()
        self.core = patch("opensv_train.capture.AcousticCore", return_value=CoreStub()).start()
        self.addCleanup(patch.stopall)
        for index, split in enumerate(("train", "validation")):
            sample = self.root / split
            sample.mkdir()
            row = dict(version=1, split=split, frames=2, phone_channels=108,
                       counts=[2], languages=[0])
            (sample / "sample.json").write_text(json.dumps(row), encoding="utf-8")
            for name, array in dict(phones=np.zeros((1, 108)), pitch=np.full((2, 1), index),
                                    noise=np.full((2, 4), index), target=np.zeros((2, 71)),
                                    baseline=np.zeros((2, 71))).items():
                array.astype("<f4").tofile(sample / f"{name}.f32")

    def test_pack_preserves_arrays_and_binds_model(self):
        pack(self.root)
        manifest = json.loads((self.root / "dataset.json").read_text())
        self.assertEqual(len(manifest["model_sha256"]), 64)
        self.assertEqual(len(manifest["samples"]), 2)
        with np.load(self.root / "validation.npz") as sample:
            self.assertTrue(np.all(sample["pitch"] == 1))
            self.assertEqual(sample["target"].shape, (2, 71))

    def test_changed_noise_does_not_hide_duplicate_inputs(self):
        np.zeros((2, 1), dtype="<f4").tofile(self.root / "validation/pitch.f32")
        with self.assertRaisesRegex(ValueError, "Repeated phrase"):
            pack(self.root)
        self.assertFalse((self.root / "dataset.json").exists())

    def test_inference_mismatch_rejects_dataset(self):
        np.ones((2, 71), dtype="<f4").tofile(self.root / "train/baseline.f32")
        with self.assertRaisesRegex(ValueError, "acoustic mismatch"):
            pack(self.root)
        self.assertFalse((self.root / "dataset.json").exists())

    def test_fractional_frame_counts_rejected(self):
        path = self.root / "train/sample.json"
        row = json.loads(path.read_text())
        row["counts"] = [2.5]
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "frame counts"):
            pack(self.root)


if __name__ == "__main__":
    unittest.main()

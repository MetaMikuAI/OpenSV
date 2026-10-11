import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from metasv_train.__main__ import load_data


class ContextStub:
    def frame_context(self, phones, counts, languages, pitch):
        return torch.zeros((len(pitch), 144))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "dataset.json"
        self.rows = []
        for index, split in enumerate(("train", "validation")):
            arrays = dict(phones=np.zeros((1, 108)), counts=np.array([2]), languages=np.array([0]),
                          pitch=np.zeros((2, 1)), noise=np.zeros((2, 4)),
                          target=np.full((2, 71), index, dtype=np.float32))
            np.savez(self.root / f"{split}.npz", **arrays)
            self.rows.append(dict(id=split, split=split, features=f"{split}.npz"))
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(dict(version=1, samples=self.rows)), encoding="utf-8")

    def test_separate_splits(self):
        groups, signatures = load_data(self.manifest, ContextStub())
        self.assertEqual(len(groups["validation"]), 1)
        self.assertNotEqual(signatures[0][2], signatures[1][2])

    def test_duplicate_content_rejected(self):
        self.rows[1]["features"] = self.rows[0]["features"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_data(self.manifest, ContextStub())

    def test_missing_validation_rejected(self):
        self.rows = self.rows[:1]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "separate"):
            load_data(self.manifest, ContextStub())

    def test_nonfinite_target_rejected(self):
        path = self.root / "train.npz"
        with np.load(path) as source:
            arrays = dict(source)
        arrays["target"][0, 0] = np.nan
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "invalid target"):
            load_data(self.manifest, ContextStub())


if __name__ == "__main__":
    unittest.main()

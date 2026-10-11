import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from opensv_train.__main__ import load_data
from opensv_train.audio import read_wav
from opensv_train.vocoder import excitation, lf_parameters, lf_spectrum


def wav(samples, rate=48000):
    raw = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    return b"RIFF" + struct.pack("<I", len(body)) + body


class AudioTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_wav_rate_and_nonfinite_rejected(self):
        path = self.root / "test.wav"
        path.write_bytes(wav([0, .5, -.5]))
        torch.testing.assert_close(read_wav(path, 48000), torch.tensor([0, .5, -.5]))
        with self.assertRaisesRegex(ValueError, "sample rate"):
            read_wav(path, 44100)
        path.write_bytes(wav([float("nan")]))
        with self.assertRaisesRegex(ValueError, "Invalid WAV"):
            read_wav(path, 48000)

    def test_audio_loader_does_not_use_hidden_targets(self):
        rows = []
        for i, split in enumerate(("train", "validation")):
            # 中间目标含 NaN，音频监督仍只读取 WAV。
            np.savez(self.root / f"{split}.npz", phones=np.zeros((1, 108)), counts=[2], languages=[0],
                     pitch=np.full((2, 1), i), log_f0=np.full((2, 1), 5), noise=np.zeros((2, 4)),
                     target=np.full((2, 71), np.nan))
            (self.root / f"{split}.wav").write_bytes(wav(np.full(480, i * .1)))
            rows.append(dict(id=split, split=split, features=f"{split}.npz", audio=f"{split}.wav"))
        path = self.root / "dataset.json"
        path.write_text(json.dumps(dict(version=1, supervision="audio", samples=rows)))
        core = SimpleNamespace(frame_context=lambda *args: torch.zeros((2, 144)))
        objective = SimpleNamespace(vocoder=SimpleNamespace(sample_rate=48000, hop=240))
        groups, _ = load_data(path, core, "acoustic", objective)
        self.assertEqual(groups["train"][0][2][1].shape, (480,))
        with self.assertRaisesRegex(ValueError, "require --vocoder"):
            load_data(path, core)
        (self.root / "validation.wav").write_bytes((self.root / "train.wav").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate target WAV"):
            load_data(path, core, "acoustic", objective)

    def test_lf_gradient_matches_finite_difference(self):
        shape = torch.tensor([.73, 1.53, 2.71], dtype=torch.float64, requires_grad=True)
        harmonic = torch.arange(1, 5, dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x: torch.view_as_real(lf_spectrum(lf_parameters(x), harmonic)),
                                                (shape,), eps=1e-5, atol=1e-5, rtol=1e-3))

    def test_excitation_gradient_matches_finite_difference(self):
        shape = torch.tensor([.73, 1.53, 2.71], requires_grad=True)
        f0 = torch.tensor([220., 220., 220.])
        def energy(x):
            return excitation(x, f0, torch.ones(3), 240, float(np.float32(.005)), .5)[0].square().mean()
        gradient, = torch.autograd.grad(energy(shape), shape)
        numeric = []
        for i in range(3):
            delta = torch.zeros(3)
            delta[i] = .001
            numeric.append((energy(shape.detach() + delta) - energy(shape.detach() - delta)) / .002)
        torch.testing.assert_close(gradient, torch.stack(numeric), atol=1e-5, rtol=.02)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from opensv_train.__main__ import main, move_sample, run, train, training_device


class SmallCore(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., device=device))

    def frame_context(self, phones, counts, languages, pitch):
        return phones + pitch

    def forward(self, inputs, noise):
        return inputs * self.weight + noise

    def export(self):
        self.weight.detach().cpu().numpy()


class TrainingDeviceTests(unittest.TestCase):
    def test_cpu_does_not_require_cuda(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
            self.assertEqual(training_device("cpu"), torch.device("cpu"))

    def test_unavailable_cuda_fails_before_loading_model(self):
        with patch("torch.cuda.is_available", return_value=False), patch("opensv_train.__main__.Model.read") as read:
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                run(SimpleNamespace(device="cuda"))
            read.assert_not_called()

    def test_sample_transfer_preserves_cpu_dataset(self):
        sample = ((torch.ones(2, 1), [2], [0], torch.zeros(2, 1)), torch.zeros(2, 1),
                  (torch.zeros(2, 1), torch.zeros(8)))
        moved = move_sample(sample, torch.device("meta"))
        self.assertEqual(moved[0][0].device.type, "meta")
        self.assertEqual(moved[2][1].device.type, "meta")
        self.assertEqual(moved[0][1], [2])
        self.assertEqual(sample[0][0].device.type, "cpu")
        self.assertEqual(sample[2][1].device.type, "cpu")

    def test_cuda_out_of_memory_has_actionable_message(self):
        argv = ["opensv-train", "--model", "base.dnni", "--dataset", "dataset.json",
                "--output", "output", "--device", "cuda"]
        with patch("sys.argv", argv), patch("opensv_train.__main__.run", side_effect=torch.cuda.OutOfMemoryError), \
                patch("opensv_train.__main__.emit") as emit:
            self.assertEqual(main(), 1)
            self.assertIn("choose CPU", emit.call_args.kwargs["message"])

    def check_resume(self, devices):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sample = ((torch.ones(2, 1), [2], [0], torch.zeros(2, 1)), torch.zeros(2, 1), torch.zeros(2, 1))
            groups = {"train": [sample], "validation": [sample]}
            state = None
            for step, device in enumerate(devices, 1):
                core = SmallCore(device)
                args = SimpleNamespace(output=output, lr=.01, scope="acoustic", steps=step,
                                       seconds=30, save_every=1)
                with patch("opensv_train.__main__.Model.read"), patch("opensv_train.__main__.emit"):
                    result = train(args, SimpleNamespace(to_bytes=lambda: b"model"), core,
                                   list(core.parameters()), groups, {}, state, time.monotonic(), random.Random(1))
                self.assertEqual(result, 0)
                state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
                self.assertEqual(state["step"], step)
                self.assertLess(state["vectors"][0].item(), 1 - .009 * step)
                self.assertEqual(state["vectors"][0].device.type, "cpu")
                self.assertEqual((output / "trained-base.dnni").read_bytes(), b"model")
                self.assertEqual(sample[0][0].device.type, "cpu")

    def test_cpu_checkpoint_resume(self):
        self.check_resume(("cpu", "cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_checkpoint_moves_between_cpu_and_cuda(self):
        self.check_resume(("cpu", "cuda", "cpu"))


if __name__ == "__main__":
    unittest.main()

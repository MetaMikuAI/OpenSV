import unittest

import numpy as np
import torch

from opensv_train.align import duration_feature, dtw_map, realigned_counts
from opensv_train.vocoder import Normalizer, mel_bank, mel_spectral_loss


def blocks(pattern, channels=8):
    """Frames whose spectrum changes abruptly between labelled sections."""
    rng = np.random.default_rng(0)
    means = rng.normal(size=(len(pattern), channels)) * 4
    return np.concatenate([np.tile(means[i], (length, 1)) for i, length in enumerate(pattern)])


class AlignTests(unittest.TestCase):
    def test_identical_material_is_left_alone(self):
        frames = blocks([40, 30, 50])
        mapping = dtw_map(frames, frames)
        np.testing.assert_array_equal(mapping, np.arange(len(frames)))
        self.assertEqual(realigned_counts([40, 30, 50], mapping, 120), ([40, 30, 50], 0.0))

    def test_moved_boundary_is_found(self):
        estimated = blocks([40, 30, 50])
        actual = blocks([50, 30, 40])  # the same sections, ten frames later in the recording
        counts, shift = realigned_counts([40, 30, 50], dtw_map(estimated, actual), 120)
        self.assertEqual(sum(counts), 120)
        self.assertLessEqual(abs(counts[0] - 50), 2)
        self.assertLessEqual(abs(counts[1] - 30), 2)
        self.assertGreater(shift, 30)

    def test_counts_stay_positive_and_keep_their_total(self):
        counts, _ = realigned_counts([1, 1, 1, 7], np.zeros(10), 10)
        self.assertEqual(sum(counts), 10)
        self.assertTrue(all(c >= 1 for c in counts))

    def test_single_phone_has_no_boundary_to_move(self):
        self.assertEqual(realigned_counts([20], np.arange(20.), 20), ([20], 0.0))

    def test_duration_feature_matches_renderer_formula(self):
        normalizer = Normalizer.__new__(Normalizer)
        torch.nn.Module.__init__(normalizer)
        normalizer.register_buffer("lower", torch.tensor([0., -3.]))
        normalizer.register_buffer("upper", torch.tensor([1., 2.]))
        normalizer.register_buffer("target", torch.tensor([-1., 1.]))
        value = duration_feature([10], normalizer)
        raw = np.log(10 * .005 + .05)
        self.assertAlmostEqual(float(value[0]), -1 + 2 * (raw + 3) / 5, places=5)


class MelLossTests(unittest.TestCase):
    def test_zero_for_identical_audio_and_differentiable(self):
        target = torch.randn(9600)
        self.assertLess(float(mel_spectral_loss(target, target)), 1e-5)
        audio = (target * .5).requires_grad_()
        loss = mel_spectral_loss(audio, target)
        loss.backward()
        self.assertGreater(float(loss), .1)
        self.assertTrue(bool(audio.grad.isfinite().all()))

    def test_ignores_content_above_the_mel_range(self):
        t = torch.arange(9600.) / 48000
        voice = torch.sin(2 * torch.pi * 440 * t)
        hiss = .01 * torch.sin(2 * torch.pi * 21000 * t)
        self.assertEqual(float(mel_bank(48000, 2048, 128)[:, 900:].sum()), 0)
        self.assertLess(float(mel_spectral_loss(voice + hiss, voice)), .05)


if __name__ == "__main__":
    unittest.main()

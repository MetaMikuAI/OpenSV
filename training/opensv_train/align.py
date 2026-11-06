"""Phone boundaries estimated from audio alone.

The score gives the phone order and rough timing. Rendering that timing with the base model and warping the result
onto the recording (dynamic time warping over log-mel frames) moves each boundary to where the recording actually
changes sound. Only the base model, the score and the recording are used.
"""
import numpy as np
import torch

from .audio import AudioObjective
from .format import Model
from .network import AcousticCore
from .vocoder import Normalizer, decode, mel_bank

FRAME_SECONDS = .005
FEATURE_FFT = 1024
FEATURE_MELS = 80
FEATURE_TOP_HZ = 16000.0


def dtw_map(model_mel, target_mel, radius=30):
    """Warp model frames onto target frames within +-radius frames; returns the target frame for each model frame."""
    a = model_mel - model_mel.mean(0, keepdims=True)
    b = target_mel - target_mel.mean(0, keepdims=True)
    frames = min(len(a), len(b))
    a, b = a[:frames], b[:frames]
    width, inf = 2 * radius + 1, 1e18
    cost = np.full((frames, width), inf)
    for i in range(frames):
        low, high = max(0, i - radius), min(frames, i + radius + 1)
        cost[i, low - i + radius:high - i + radius] = np.abs(a[i] - b[low:high]).mean(1)
    total = np.full_like(cost, inf)
    move = np.zeros(cost.shape, np.int8)
    total[0, radius] = cost[0, radius]
    for j in range(radius + 1, width):
        if cost[0, j] < inf:
            total[0, j], move[0, j] = total[0, j - 1] + cost[0, j] * 1.05, 1
    # Steps off the diagonal carry a small penalty, so unchanged timing stays exactly unchanged.
    for i in range(1, frames):
        previous, row_cost, row, row_move = total[i - 1], cost[i], total[i], move[i]
        for j in range(width):
            step = row_cost[j]
            if step >= inf:
                continue
            best, choice = previous[j], 0
            if j + 1 < width and previous[j + 1] + step * .05 < best:
                best, choice = previous[j + 1] + step * .05, 2
            if j > 0 and row[j - 1] + step * .05 < best:
                best, choice = row[j - 1] + step * .05, 1
            row[j], row_move[j] = best + step, choice
    i, j = frames - 1, radius
    hits = [[] for _ in range(frames)]
    while i >= 0:
        hits[i].append(i + j - radius)
        choice = move[i, j]
        if choice == 0:
            i -= 1
        elif choice == 2:
            i, j = i - 1, j + 1
        else:
            j -= 1
            if j < 0:
                break
    return np.array([np.mean(h) if h else k for k, h in enumerate(hits)])


def realigned_counts(counts, mapping, frames):
    """Move phone boundaries through the warp. Every phone keeps at least one frame and the total stays the same."""
    edges = np.cumsum(counts)[:-1]
    if not len(edges):
        return list(counts), 0.0
    moved = np.rint(mapping[np.minimum(edges, len(mapping) - 1)]).astype(int)
    for k in range(len(moved)):
        moved[k] = max(moved[k], (moved[k - 1] if k else 0) + 1)
    for k in range(len(moved) - 1, -1, -1):
        moved[k] = min(moved[k], (moved[k + 1] if k + 1 < len(moved) else frames) - 1)
    result = np.diff(np.concatenate(([0], moved, [frames])))
    if (result < 1).any() or result.sum() != frames:
        return list(counts), 0.0
    return result.tolist(), float(np.abs(moved - edges).mean() * FRAME_SECONDS * 1000)


def duration_feature(counts, normalizer):
    """The phone encoder's last input channel: normalized log duration, as computed by the renderer."""
    raw = torch.log(torch.tensor(counts, dtype=torch.float32) * FRAME_SECONDS + .05)
    low, high, goal = normalizer.lower[-1].cpu(), normalizer.upper[-1].cpu(), normalizer.target.cpu()
    return goal[0] + (goal[1] - goal[0]) * (raw - low) / (high - low).clamp_min(1e-5)


class Aligner:
    def __init__(self, model_path, vocoder_path, device="cpu"):
        model = Model.read(model_path)
        self.device = torch.device(device)
        self.core = AcousticCore(model.root).eval().requires_grad_(False).to(self.device)
        self.objective = AudioObjective(model, vocoder_path).eval().to(self.device)
        features = next(n for n in model.nodes if n.type in ("_ftmfv2", "_ftmfv3"))
        self.phones = Normalizer(features.children[3])
        self.rate = self.objective.vocoder.sample_rate
        self.hop = self.objective.vocoder.hop
        self.bank = mel_bank(self.rate, FEATURE_FFT, FEATURE_MELS, high=FEATURE_TOP_HZ, device=self.device)
        self.window = torch.hann_window(FEATURE_FFT, device=self.device)

    def log_mel(self, signal):
        spectrum = torch.stft(signal, FEATURE_FFT, self.hop, window=self.window, return_complex=True,
                              pad_mode="constant").abs().square()
        return (self.bank @ spectrum).T.clamp_min(1e-8).log().cpu().numpy()

    @torch.no_grad()
    def align(self, arrays, waveform):
        """Return (counts, phones, mean boundary shift in ms) for one prepared segment."""
        device = self.device
        counts = np.asarray(arrays["counts"]).tolist()
        phones = torch.from_numpy(np.asarray(arrays["phones"], np.float32)).to(device)
        log_f0 = torch.from_numpy(np.asarray(arrays["log_f0"], np.float32)).to(device)
        context = self.core.frame_context(phones, counts, np.asarray(arrays["languages"]).tolist(),
                                          torch.from_numpy(np.asarray(arrays["pitch"], np.float32)).to(device))
        prediction = self.core(context, torch.from_numpy(np.asarray(arrays["noise"], np.float32)).to(device))
        audio = self.objective.vocoder(decode(self.objective.normalizer, prediction, log_f0))
        frames = len(log_f0)
        mapping = dtw_map(self.log_mel(audio)[:frames], self.log_mel(waveform.to(device))[:frames])
        moved, shift = realigned_counts(counts, mapping, frames)
        updated = np.asarray(arrays["phones"], np.float32).copy()
        updated[:, -1] = duration_feature(moved, self.phones).numpy()
        return np.asarray(moved, np.int64), updated, shift

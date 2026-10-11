"""WAV 监督与可微分合成。"""

import struct
from pathlib import Path

import numpy as np
import torch

from .format import Model
from .runtime import digest
from .vocoder import Normalizer, Vocoder, decode, spectral_loss


def read_wav(path, sample_rate):
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected RIFF WAV")
    position, fmt, samples = 12, None, None
    while position + 8 <= len(data):
        tag, size = struct.unpack_from("<4sI", data, position)
        start = position + 8
        if start + size > len(data):
            raise ValueError("Truncated WAV")
        if tag == b"fmt ":
            if size < 16:
                raise ValueError("Invalid WAV format")
            fmt = struct.unpack_from("<HHIIHH", data, start)
            if fmt[0] == 0xFFFE:
                if size < 40 or data[start + 28:start + 40] != bytes.fromhex("00001000800000aa00389b71"):
                    raise ValueError("Unsupported extensible WAV format")
                fmt = (struct.unpack_from("<I", data, start + 24)[0], *fmt[1:])
        elif tag == b"data":
            samples = data[start:start + size]
        position = start + size + (size % 2)
    if fmt is None or samples is None or fmt[1] != 1 or fmt[2] != sample_rate:
        raise ValueError("WAV must be mono with the vocoder sample rate")
    if fmt[0] == 3 and fmt[5] == 32:
        result = np.frombuffer(samples, dtype="<f4").copy()
    elif fmt[0] == 1 and fmt[5] == 16:
        result = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768
    elif fmt[0] == 1 and fmt[5] == 24:
        b = np.frombuffer(samples, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        result = ((value ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    else:
        raise ValueError("Supported WAV: float32 or PCM16/24")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Invalid WAV samples")
    return torch.from_numpy(result)


class AudioObjective:
    def __init__(self, model, path):
        features = next(n for n in model.nodes if n.type in ("_ftmfv2", "_ftmfv3"))
        self.normalizer = Normalizer(features.children[5])
        self.vocoder = Vocoder(Model.read(path).root)
        self.signature = digest(path)

    def __call__(self, prediction, target):
        log_f0, waveform = target
        audio = self.vocoder(decode(self.normalizer, prediction, log_f0))
        if audio.shape != waveform.shape:
            raise ValueError("WAV length does not match aligned frames")
        return spectral_loss(audio, waveform)

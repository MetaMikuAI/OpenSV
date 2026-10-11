"""可微分声码器，保持推理时的滤波与激励结构。"""

import math
import struct
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .format import tensor
from .network import Network


class Normalizer(nn.Module):
    def __init__(self, node):
        super().__init__()
        if node.type != "cmpu0" or len(node.children) != 3:
            raise ValueError("Unsupported normalizer")
        for name, child in zip(("lower", "upper", "target"), node.children):
            self.register_buffer(name, torch.from_numpy(tensor(child)))

    def forward(self, x, inverse=False, clamp=False):
        width = self.target[1] - self.target[0]
        if inverse:
            return self.lower + (self.upper - self.lower) * (x - self.target[0]) / width
        y = self.target[0] + width * (x - self.lower) / (self.upper - self.lower).clamp_min(1e-5)
        return y.clamp(self.target[0], self.target[1]) if clamp else y


def decode(normalizer, features, log_f0):
    x = normalizer(features, inverse=True)
    return torch.cat(((x[:, :1] > .5).to(x.dtype), log_f0.exp(),
                      x[:, 1:2].clamp(.01, 3), x[:, 2:66], x[:, 66:].clamp(0, 1)), 1)


@lru_cache(maxsize=1)
def knots():
    values = []
    for i in range(256):
        rd = .01 * 600 ** (i / 255)
        ra = 1e-6 if rd < .21 else (4.8 * rd - 1) / 100 if rd < 2.7 else .323 / rd
        if i < 209:
            rk = (22.4 + 11.8 * rd) / 100
            rg = .25 * rk / (.11 * rd / (.5 + 1.2 * rk) - ra)
        else:
            oq = 1 - 1 / (2.17 * rd)
            rg = .0093552 + 5.96 / (7.92 - 2 * oq)
            rk = 2 * rg * oq - .9572
        peak = .5 / rg
        closing = peak * (1 + rk)
        tail = 1 - closing
        lo, hi = 0, 1 / ra
        for _ in range(80):
            d = (lo + hi) / 2
            if -math.expm1(-tail * d) / d > ra:
                lo = d
            else:
                hi = d
        decay = (lo + hi) / 2
        area = (-math.expm1(-tail * decay) - tail * decay * math.exp(-tail * decay)) / (decay ** 2 * ra)
        w = math.pi / peak
        s, c = math.sin(w * closing), math.cos(w * closing)
        def residual(g):
            return area * s * (g * g + w * w) + g * s - w * c + w * math.exp(-g * closing)
        lo, hi = 0, 1
        while residual(hi) > 0:
            hi *= 2
        for _ in range(80):
            g = (lo + hi) / 2
            if residual(g) > 0:
                lo = g
            else:
                hi = g
        values.append((rd, closing, peak, ra, math.log(decay / .005), math.log((lo + hi) / .01)))
    return torch.tensor(values, dtype=torch.float32)


def lf_parameters(shape):
    table = knots().to(shape.device)
    upper = torch.searchsorted(table[:, 0].contiguous(), shape.contiguous(), right=True)
    left, right = table[(upper - 1).clamp(0, 255)], table[upper.clamp(0, 255)]
    t = ((shape.double() - left[:, 0].double()) /
         (right[:, 0] - left[:, 0]).double().clamp_min(1e-20))[:, None]
    p = left[:, 1:].double() * (1 - t) + right[:, 1:].double() * t
    return torch.cat((p[:, :3], p[:, 3:].exp() * .005), 1)


def lf_spectrum(p, harmonic):
    closing, peak, ra, decay, growth = (v[:, None] for v in p.unbind(1))
    w = 2 * math.pi * harmonic
    sw = math.pi / peak
    angle = sw * closing
    sine = angle.sin()
    s = growth - 1j * w
    rotation = torch.exp(-1j * w * closing)
    opening = -(rotation * (s * sine - sw * angle.cos()) + sw * (-growth * closing).exp()) / (sine * (s * s - sw * sw))
    rate = decay + 1j * w
    endpoint = 1 - decay * ra
    extra = rotation * endpoint * (1 - torch.exp(-1j * w * (1 - closing))) / (1j * w * ra * rate)
    return opening - rotation / rate + torch.where(decay * endpoint >= .005, extra, 0)


def overlap(signal, hop, offset, length):
    indices = torch.arange(signal.shape[0], device=signal.device)[:, None] * hop
    indices = indices + torch.arange(signal.shape[1], device=signal.device) - offset
    valid = (indices >= 0) & (indices < length)
    return signal.new_zeros(length).scatter_add(0, indices[valid], signal[valid])


def excitation(shape, f0, voiced, hop, period, gain):
    width = 2 * hop
    n = 1 << (2 * width - 2).bit_length()
    rate = np.float32(hop) / np.float32(period)
    increment = (f0 * period).double()
    phases = increment.cumsum(0)
    count = (.5 * float(rate) / f0).long().clamp(max=width)
    p = lf_parameters(shape)
    h = torch.arange(1, width, device=f0.device, dtype=torch.float64)
    spectrum = lf_spectrum(p, h)
    reference = spectrum[:, 0].angle()
    scale = .1 * float(rate) / (f0.double() * count.clamp_min(1).double() * spectrum[:, 1].abs())
    phase = 2 * math.pi * (phases.remainder(1) - increment)
    i = torch.arange(width, device=f0.device, dtype=torch.float64)
    chirp = torch.exp(1j * math.pi * f0.double()[:, None] / float(rate) * i.square())
    a = spectrum * torch.exp(1j * h * (phase - reference)[:, None]) * chirp[:, 1:]
    a = a * (h <= count[:, None])
    a = F.pad(a.to(torch.complex64), (1, n - width))
    b = F.pad(chirp.conj().to(torch.complex64), (0, n - width))
    # 负序列仅保留参与卷积的谐波范围。
    tail = (chirp[:, 1:].conj() * (h <= count[:, None])).flip(1).to(torch.complex64)
    b = b + F.pad(tail, (n - width + 1, 0))
    signal = (torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b))[:, :width].to(torch.complex128) * chirp).real
    window = gain * (.5 - .5 * torch.cos(2 * math.pi * i / width))
    source = overlap((signal * (scale * voiced)[:, None] * window).float(), hop, hop, len(f0) * hop)
    return source, phases


@lru_cache(maxsize=16)
def gaussian_noise(frames, bands):
    state, spare = 1, None
    result = []
    f = np.float32
    def uniform():
        nonlocal state
        state = state * 48271 % 2147483647
        return f(f(state - 1) * f(1 / 1073741824) - f(1))
    for _ in range(frames * bands):
        if spare is not None:
            result.append(spare)
            spare = None
            continue
        while True:
            a, b = uniform(), uniform()
            radius = f(a * a + b * b)
            if 0 < radius <= 1:
                break
        scale = f(np.sqrt(f(f(-2) * np.log(radius) / radius)))
        result.append(f(a * scale))
        spare = f(b * scale)
    return torch.tensor(np.asarray(result).reshape(frames, bands))


class Vocoder(nn.Module):
    def __init__(self, root):
        super().__init__()
        c = root.children
        if len(c) != 13 or len(root.payload) != 18 or root.payload[12] != 1:
            raise ValueError("Unsupported vocoder")
        config = c[0]
        if ((config.type, len(config.payload)) not in (("_vocfv1", 48), ("_vocfv2", 52))
                or struct.unpack_from("<I", config.payload)[0] != 72):
            raise ValueError("Unsupported vocoder configuration")
        self.sample_rate, self.period = struct.unpack_from("<If", config.payload, 8)
        if self.sample_rate <= 0 or not math.isfinite(self.period) or self.period <= 0:
            raise ValueError("Invalid vocoder timing")
        self.hop = round(self.sample_rate * self.period)
        if not 8 <= self.hop <= 65536:
            raise ValueError("Invalid vocoder hop")
        self.force_voiced = len(config.payload) == 52 and struct.unpack_from("<I", config.payload, 48)[0] != 0
        self.gain = float(np.exp(np.float32(struct.unpack_from("<f", root.payload, 4)[0])))
        self.bias = struct.unpack_from("<f", root.payload, 14)[0]
        self.gate = bool(root.payload[13])
        self.normalizer = Normalizer(config.children[0])
        self.shape = Network(c[1])
        self.condition = Network(c[2])
        self.modulation = Network(c[5])
        self.residual = Network(c[11])
        self.projection = Network(c[12])
        if any(len(node.payload) != 13 or struct.unpack_from("<I", node.payload)[0] != 2 for node in c[6:9]):
            raise ValueError("Unsupported spectral filter")
        if len(c[3].children) != 1 or len(c[3].payload) < 8:
            raise ValueError("Invalid spectral heads")
        streams, count = struct.unpack_from("<II", c[3].payload)
        if streams != 3 or not count or len(c[3].payload) != 8 + 16 * count or len(c[3].children[0].children) != count:
            raise ValueError("Invalid spectral head dimensions")
        self.heads = nn.ModuleList(Network(node) for node in c[3].children[0].children)
        self.bands = [struct.unpack_from("<IIII", c[3].payload, 8 + 16 * i) for i in range(len(self.heads))]
        self.bins = max(b[1] for b in self.bands)
        for first, end, fade_in, fade_out in self.bands:
            if not 0 <= first < end <= 65536 or 1 in (fade_in, fade_out) or fade_in + fade_out > end - first:
                raise ValueError("Invalid spectral band")
        if self.bins < 4 or self.bins % 2 or self.bins > 1 << (4 * self.hop - 2).bit_length():
            raise ValueError("Invalid spectrum width")
        for name, value in (("ratios", tensor(c[4])),
                            ("synthesis", np.stack([tensor(node) for node in c[9].children])),
                            ("analysis", np.stack([tensor(node) for node in c[10].children]))):
            self.register_buffer(name, torch.from_numpy(value))
        if (self.analysis.ndim != 2 or self.synthesis.ndim != 2
                or len(self.analysis) != len(self.synthesis) or len(self.analysis) < 2
                or self.hop % len(self.analysis) or self.analysis.shape[1] % 2 != 1
                or self.synthesis.shape[1] % 2 != 1):
            raise ValueError("Invalid subband filters")
        self.requires_grad_(False)

    def noise(self, x, source):
        bands, width = self.analysis.shape
        factor = self.hop // bands
        frames = len(x) * factor
        first = bands - 1 - (width // 2) % bands
        indices = first + (torch.arange(frames, device=x.device)[:, None] + bands // 2 - 1) * bands
        indices = indices - (width - 1) + torch.arange(width, device=x.device)
        signals = source[indices.clamp(0, len(source) - 1)] * ((indices >= 0) & (indices < len(source)))
        noise = gaussian_noise(frames, bands).to(x.device)
        inputs = torch.cat((signals @ self.analysis.T, noise), 1)
        positions = (torch.arange(frames, device=x.device) - factor // 2 + (.5 if factor % 2 == 0 else 0)) / factor
        left = positions.floor().long()
        mix = (positions - left)[:, None]
        condition = x[left.clamp(0, len(x) - 1)] * (1 - mix) + x[(left + 1).clamp(0, len(x) - 1)] * mix
        y = self.projection(self.residual(inputs, condition))
        if self.gate:
            y = noise * y.sigmoid()
        filtered = y @ self.synthesis
        latency = (self.synthesis.shape[1] - 1) // 2
        indices = (torch.arange(frames, device=x.device) * bands - latency).clamp_min(0)[:, None]
        indices = indices + torch.arange(self.synthesis.shape[1], device=x.device)
        valid = indices < len(source)
        return source.new_zeros(len(source)).scatter_add(0, indices[valid], filtered[valid])

    def filter(self, values, source):
        n = 1 << (4 * self.hop - 2).bit_length()
        half = values.shape[1] // 2
        first = 10 * values[:, :half + 1] / torch.arange(1, half + 2, device=values.device)
        last = 10 * values[:, half + 1:] / torch.arange(half, 1, -1, device=values.device)
        cepstrum = torch.cat((first, values.new_zeros(len(values), n - 2 * half), last), 1)
        response = torch.fft.rfft(cepstrum).exp()
        indices = torch.arange(len(values), device=values.device)[:, None] * self.hop
        indices = indices + torch.arange(2 * self.hop, device=values.device) - self.hop
        window = torch.hann_window(2 * self.hop, device=values.device)
        signal = source[indices.clamp(0, len(source) - 1)] * ((indices >= 0) & (indices < len(source))) * window
        signal = F.pad(signal, ((n - 2 * self.hop) // 2,) * 2)
        filtered = torch.fft.irfft(torch.fft.rfft(signal) * response, n=n)
        edge = values.new_ones(n)
        edge[:16] = torch.arange(1, 17, device=values.device) / 16
        edge[-16:] = torch.arange(15, -1, -1, device=values.device) / 16
        return overlap(filtered * edge, self.hop, n // 2, len(source))

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != 72 or not 0 < len(inputs) <= 32768:
            raise ValueError("Invalid vocoder input shape")
        if not bool(inputs.isfinite().all()) or not bool((inputs[:, 1] > 0).all()):
            raise ValueError("Invalid vocoder input values")
        voiced = torch.ones_like(inputs[:, 0]) if self.force_voiced else (inputs[:, 0] >= .5).float()
        f0 = inputs[:, 1]
        x = self.normalizer(torch.cat((voiced[:, None], f0.log()[:, None], inputs[:, 2:]), 1), clamp=True)
        shape = 2.9 * self.shape(x).sigmoid()[:, 0] + .1
        condition = self.condition(x)
        spectra = x.new_zeros(len(x), 3, self.bins)
        for network, (first, end, fade_in, fade_out) in zip(self.heads, self.bands):
            width = end - first
            weights = x.new_ones(width)
            if fade_in:
                weights[:fade_in] = torch.arange(fade_in, device=x.device) / (fade_in - 1)
            if fade_out:
                weights[-fade_out:] = torch.arange(fade_out - 1, -1, -1, device=x.device) / (fade_out - 1)
            spectra = spectra + F.pad(network(condition).reshape(len(x), 3, width) * weights, (first, self.bins - end))
        source, phases = excitation(shape, f0, voiced, self.hop, self.period, self.gain)
        parameters = self.modulation(condition).double()
        ratios = self.ratios.double()
        count = len(ratios)
        angular = 2 * math.pi * ratios
        phase = phases.float().double()[:, None] * angular + parameters[:, count:]
        i = torch.arange(2 * self.hop, device=x.device, dtype=torch.float64)
        step = angular * f0.double()[:, None] * self.period / self.hop
        signal = (parameters[:, :count].exp()[:, :, None] * (phase[:, :, None] + (i - self.hop) * step[:, :, None]).cos()).sum(1)
        window = .5 - .5 * (2 * math.pi * i / (2 * self.hop)).cos()
        modulated = source * (overlap((signal * window).float(), self.hop, self.hop, len(source)) + self.bias)
        residual = self.noise(x, source)
        return sum(self.filter(spectra[:, i], signal) for i, signal in enumerate((source, residual, modulated)))


def spectral_loss(audio, target):
    losses = []
    for size in (256, 512, 1024):
        window = torch.hann_window(size, device=audio.device)
        def magnitude(x):
            return torch.stft(x, size, size // 4, window=window, return_complex=True, pad_mode="constant").abs().clamp_min(1e-5)
        actual, expected = magnitude(audio), magnitude(target)
        losses.append((actual - expected).norm() / expected.norm().clamp_min(1e-5)
                      + (actual.log() - expected.log()).abs().mean())
    return torch.stack(losses).mean()

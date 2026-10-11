"""可求导的 DNNI 声学网络。"""

import struct

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .format import MATRICES, replace_tensor, tensor


class Network(nn.Module):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.kind = node.type
        self.parts = nn.ModuleList()
        self.weights = nn.ParameterList()
        self.tensor_nodes = []
        self.bias = None
        self.bias_node = None
        self.alpha = 0.0
        p, children = node.payload, node.children

        def require(condition, message):
            if not condition:
                raise ValueError(f"Node {node.index} ({node.type}): {message}")

        def weight(child):
            require(child.type in MATRICES and not child.children, "expected matrix leaf")
            self.weights.append(nn.Parameter(torch.from_numpy(tensor(child))))
            self.tensor_nodes.append(child)

        if self.kind in ("modm0", "modl6"):
            require(not p, "unexpected payload")
            if self.kind == "modl6":
                require(len(children) == 2 and all(n.type == "modl3" for n in children), "expected two GRUs")
            self.parts.extend(Network(c) for c in children)
        elif self.kind in ("modl0", "modl1"):
            self.stride, self.padding, self.dilation = 1, 0, 1
            if self.kind == "modl1":
                require(len(p) == 20, "expected Conv1D payload")
                kernel, self.stride, self.padding, self.dilation, groups = struct.unpack("<5i", p)
                require(kernel > 0 and self.stride > 0 and self.padding >= 0
                        and self.dilation > 0 and groups == 1, "unsupported convolution")
            else:
                require(not p, "unexpected dense payload")
                kernel = 1
            for c in children:
                if c.type == "prim1":
                    require(self.bias is None and not c.children, "invalid bias")
                    self.bias = nn.Parameter(torch.from_numpy(tensor(c)))
                    self.bias_node = c
                else:
                    weight(c)
            require(len(self.weights) == kernel, "kernel count mismatch")
            require(all(w.shape == self.weights[0].shape for w in self.weights), "kernel shape mismatch")
            require(self.bias is None or self.bias.numel() == self.weights[0].shape[0], "bias shape mismatch")
        elif self.kind == "modl3":
            require(not p and len(children) == 12, "expected six matrix/bias pairs")
            self.gru_biases = nn.ParameterList()
            for i in range(6):
                weight(children[2 * i])
                c = children[2 * i + 1]
                require(c.type == "prim1" and not c.children, "expected GRU bias")
                self.gru_biases.append(nn.Parameter(torch.from_numpy(tensor(c))))
            hidden, inputs = self.weights[0].shape
            require(all(tuple(w.shape) == (hidden, inputs if i < 3 else hidden)
                        for i, w in enumerate(self.weights)), "GRU matrix shape mismatch")
            require(all(b.numel() == hidden for b in self.gru_biases), "GRU bias shape mismatch")
        elif self.kind in ("_gnc1v0", "_ncwnv0"):
            require(len(p) == 12, "expected channel dimensions")
            self.input_channels, self.hidden_channels, self.condition_channels = struct.unpack("<3i", p)
            require(self.input_channels > 0 and self.hidden_channels > 0 and self.condition_channels >= 0,
                    "invalid channels")
            if self.kind == "_gnc1v0":
                require(len(children) == (2 if self.condition_channels else 1)
                        and all(c.type == "modl1" for c in children), "expected gate convolutions")
                self.parts.extend(Network(c) for c in children)
                for i, part in enumerate(self.parts):
                    expected_input = self.input_channels if i == 0 else self.condition_channels
                    require(tuple(part.weights[0].shape) == (2 * self.hidden_channels, expected_input),
                            "gate projection dimensions mismatch")
                    require(i == 0 or len(part.weights) == 1, "condition kernel must be one frame")
            else:
                require(len(children) == (4 if self.condition_channels else 3), "invalid residual children")
                groups = children[:3]
                require(all(c.type == "cmpg1" and not c.payload for c in groups), "expected residual groups")
                self.stages = len(groups[0].children)
                require(self.stages > 0 and all(len(c.children) == self.stages for c in groups), "invalid stages")
                for i in range(self.stages):
                    for j, group in enumerate(groups):
                        require(group.children[i].type == ("_gnc1v0" if j == 0 else "modl1"),
                                "invalid residual stage type")
                        self.parts.append(Network(group.children[i]))
                    gate, skip, residual = self.parts[-3:]
                    stage_input = self.input_channels if i == 0 else self.hidden_channels
                    require((gate.input_channels, gate.hidden_channels, gate.condition_channels) ==
                            (stage_input, self.hidden_channels, self.condition_channels), "gate dimensions mismatch")
                    require(tuple(skip.weights[0].shape) == (self.hidden_channels, self.hidden_channels)
                            and tuple(residual.weights[0].shape) == (self.hidden_channels, stage_input),
                            "residual projection dimensions mismatch")
                    for part in [*gate.parts, skip, residual]:
                        require(part.stride == 1 and 2 * part.padding == (len(part.weights) - 1) * part.dilation,
                                "residual stage must preserve frames")
                if self.condition_channels:
                    require(children[3].type == "modl1", "condition projection must be Conv1D")
                    self.parts.append(Network(children[3]))
                    part = self.parts[-1]
                    require(tuple(part.weights[0].shape) == (self.hidden_channels, self.condition_channels)
                            and len(part.weights) == 1 and part.stride == 1 and part.padding == 0,
                            "condition projection dimensions/timing mismatch")
        elif self.kind in ("moda0", "moda1", "moda2", "moda3", "moda4", "moda5", "moda7"):
            require(not children, "activation has children")
            if self.kind in ("moda3", "moda4"):
                require(len(p) == 4, "expected alpha")
                self.alpha, = struct.unpack("<f", p)
                require(np.isfinite(self.alpha), "non-finite alpha")
            else:
                require(not p, "unexpected activation payload")
        else:
            raise ValueError(f"Unsupported operator {self.kind} at node {node.index}")

    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    def gru(self, x, reverse=False):
        h = x.new_zeros(self.weights[0].shape[0])
        output = []
        for frame in (x.flip(0) if reverse else x):
            q = [F.linear(frame if i < 3 else h, w) for i, w in enumerate(self.weights)]
            b = self.gru_biases
            reset = self.sigmoid(b[0] + b[3] + q[0] + q[3])
            update = self.sigmoid(b[1] + b[4] + q[1] + q[4])
            candidate = torch.tanh(reset * (b[5] + q[5]) + q[2] + b[2])
            h = update * h + (1 - update) * candidate
            output.append(h)
        y = torch.stack(output) if output else x.new_empty((0, h.numel()))
        return y.flip(0) if reverse else y

    def forward(self, x, condition=None):
        if x.ndim != 2:
            raise ValueError("Expected frame-major [frames, channels]")
        k = self.kind
        if k == "modm0":
            for part in self.parts:
                x = part(x, condition)
            return x
        if k == "modl0":
            return F.linear(x, self.weights[0], self.bias)
        if k == "modl1":
            kernel = torch.stack(list(self.weights), dim=2)
            if x.shape[0] + 2 * self.padding < self.dilation * (kernel.shape[2] - 1) + 1:
                return x.new_empty((0, kernel.shape[0]))
            return F.conv1d(x.T.unsqueeze(0), kernel, self.bias, self.stride,
                            self.padding, self.dilation).squeeze(0).T
        if k == "modl3":
            return self.gru(x)
        if k == "modl6":
            return torch.cat((self.parts[0].gru(x), self.parts[1].gru(x, True)), dim=1)
        if k in ("_gnc1v0", "_ncwnv0"):
            if x.shape[1] != self.input_channels:
                raise ValueError("Input channel mismatch")
            if self.condition_channels and (condition is None or condition.ndim != 2
                    or condition.shape[1] != self.condition_channels):
                raise ValueError("Missing or mismatched condition tensor")
        if k == "_gnc1v0":
            y = self.parts[0](x)
            if self.condition_channels:
                projected = self.parts[1](condition)
                if projected.shape != y.shape:
                    raise ValueError("Condition frame count mismatch")
                y = y + projected
            if y.shape[1] != 2 * self.hidden_channels:
                raise ValueError("Gate output shape mismatch")
            a, b = y.chunk(2, dim=1)
            return torch.tanh(a) * self.sigmoid(b)
        if k == "_ncwnv0":
            y = self.parts[-1](condition) if self.condition_channels else x.new_zeros((x.shape[0], self.hidden_channels))
            for i in range(self.stages):
                gate, skip, residual = self.parts[i * 3:i * 3 + 3]
                g = gate(x, condition)
                r, s = residual(x), skip(g)
                if y.shape != g.shape or y.shape != r.shape or y.shape != s.shape:
                    raise ValueError("Residual branch shape mismatch")
                y = y + s
                x = g + r
            return y
        if k == "moda0":
            return x.clamp_min(0)
        if k == "moda1":
            return torch.tanh(x)
        if k == "moda2":
            return self.sigmoid(x)
        if k == "moda3":
            return torch.where(x > 0, x, self.alpha * x)
        if k == "moda4":
            if self.alpha < 0:
                return x.clamp_min(0) + (self.alpha * torch.expm1(x)).clamp_max(0)
            return x.clamp_min(0) + (self.alpha * torch.expm1(x.clamp_max(0))).clamp_max(0)
        if k == "moda5":
            return x
        if k == "moda7":
            return x * self.sigmoid(x)
        raise ValueError(k)

    def export(self):
        """写回参数，未修改的张量保留原始字节。"""
        for node, value in zip(self.tensor_nodes, self.weights):
            replace_tensor(node, value.detach().cpu().numpy())
        if self.bias is not None:
            replace_tensor(self.bias_node, self.bias.detach().cpu().numpy())
        if self.kind == "modl3":
            for i, value in enumerate(self.gru_biases):
                replace_tensor(self.node.children[2 * i + 1], value.detach().cpu().numpy())
        for part in self.parts:
            part.export()


class AcousticCore(nn.Module):
    """十个声学子网络，输出 71 维归一化特征，噪声由调用方提供。"""

    def __init__(self, root):
        super().__init__()
        if root.type != "_rldtg0" or len(root.children) != 2:
            raise ValueError("Expected _rldtg0")
        model, styles = root.children
        if (model.type != "_rldms0" or len(model.children) != 8
                or styles.type != "_stbkv1" or len(styles.children) != 2):
            raise ValueError("Unsupported acoustic layout")
        context, latent = model.children[2:4]
        if (context.type != "_vqctx1" or len(context.children) != 4
                or latent.type != "_didsv0" or len(latent.children) != 3
                or latent.payload != struct.pack("<IIf", 4, 64, 64.0)):
            raise ValueError("Unsupported context/latent layout")
        if (context.children[3].type != "modl4" or len(context.children[3].children) != 1
                or styles.children[0].type != "cmpg1" or not styles.children[0].children):
            raise ValueError("Unsupported embedding/style layout")
        names = ("phoneme_encoder", "context_projection", "context_network",
                 "latent_projection", "latent_network", "latent_head",
                 "spectral_network", "spectral_head", "auxiliary_network", "auxiliary_head")
        nodes = context.children[:3] + latent.children[:3] + model.children[4:8]
        self.networks = nn.ModuleDict({name: Network(node) for name, node in zip(names, nodes)})
        self.speaker_node = model.children[1]
        self.style_node = styles.children[0].children[0]
        self.language_node = context.children[3].children[0]
        self.speaker = nn.Parameter(torch.from_numpy(tensor(self.speaker_node)))
        self.style = nn.Parameter(torch.from_numpy(tensor(self.style_node)))
        self.language = nn.Parameter(torch.from_numpy(tensor(self.language_node)))
        if self.speaker.shape != (32,) or self.style.shape != (32,) or self.language.shape[0] != 8:
            raise ValueError("Unsupported speaker/style/language dimensions")

    def frame_context(self, normalized_phones, counts, language_indices, normalized_pitch):
        encoded = self.networks["phoneme_encoder"](normalized_phones)
        if encoded.shape != (len(counts), 64) or self.speaker.numel() != 32 or self.style.numel() != 32:
            raise ValueError("Unsupported context dimensions")
        if any(count < 1 for count in counts) or sum(counts) != normalized_pitch.numel():
            raise ValueError("Invalid frame counts")
        if (len(language_indices) != len(counts)
                or any(i < 0 or i >= self.language.shape[1] for i in language_indices)):
            raise ValueError("Invalid language indices")
        positions = torch.cat([torch.linspace(0, 1, n, device=encoded.device, dtype=encoded.dtype)
                               if n > 1 else encoded.new_tensor([0.5]) for n in counts])
        counts_tensor = torch.tensor(counts, device=encoded.device)
        repeated = torch.repeat_interleave(encoded, counts_tensor, dim=0)
        language = self.language[:, language_indices].T.repeat_interleave(counts_tensor, dim=0)
        phases = encoded.new_tensor([1.5707963705062866, 0, -1.5707963705062866, -3.1415927410125732])
        scalars = torch.stack((positions, normalized_pitch.flatten()), dim=1)
        periodic = (torch.cos(scalars[:, :, None] * 6.2831854820251465 + phases) * 0.5 + 0.5).flatten(1)
        frames = sum(counts)
        return torch.cat((repeated, periodic, self.speaker.expand(frames, -1), language,
                          self.style.expand(frames, -1)), dim=1)

    def training_parameters(self):
        """最终残差仅写入未返回的状态，不参与输出损失。"""
        self.requires_grad_(True)
        for module in self.modules():
            if isinstance(module, Network) and module.kind == "_ncwnv0":
                module.parts[module.stages * 3 - 1].requires_grad_(False)
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, frame_context, noise):
        if frame_context.shape[1] != 144 or noise.shape != (frame_context.shape[0], 4):
            raise ValueError("Expected context [T,144], explicit standard Gaussian noise [T,4]")
        n = self.networks
        context = n["context_network"](frame_context)
        projected = n["context_projection"](frame_context)
        latent_input = n["latent_projection"](projected)
        index = torch.arange(32, device=context.device, dtype=context.dtype)
        frequency = torch.exp2(index * (-26.575424194335938 / 64))
        diffusion = 64 * 0.995 * frequency
        guidance = 64 * frequency
        controls = torch.cat((diffusion.sin(), diffusion.cos(), guidance.sin(), guidance.cos()))
        condition = torch.cat((context, controls.expand(context.shape[0], -1), noise * 0.5), dim=1)
        latent = n["latent_head"](n["latent_network"](latent_input, condition))
        spectrum = n["spectral_head"](n["spectral_network"](context, latent))
        auxiliary = n["auxiliary_head"](n["auxiliary_network"](context, spectrum))
        return torch.cat((auxiliary[:, :2], spectrum, auxiliary[:, 2:]), dim=1)

    def export(self):
        for module in self.networks.values():
            module.export()
        for node, param in ((self.speaker_node, self.speaker), (self.style_node, self.style),
                            (self.language_node, self.language)):
            replace_tensor(node, param.detach().cpu().numpy())

"""基于渲染中间特征的分模块蒸馏。"""
import argparse
import json
from pathlib import Path
import struct
import time
import hashlib
import shutil

import numpy as np
import torch
from torch import nn

from .runtime import atomic_save, digest, emit
from .format import Model, replace_tensor, tensor
from .network import Network

FILES = {"duration": "timing_model.dnni", "pitch": "f0_model.dnni", "vocoder": "pitch_model.dnni"}
OPERATORS = {"modm0", "modl0", "modl1", "modl3", "modl6", "_gnc1v0", "_ncwnv0"}


def at(model, path):
    node = model.root
    for index in path:
        node = node.children[index]
    return node


def roots(node, path=()):
    if node.type in OPERATORS:
        yield path, node
    else:
        for index, child in enumerate(node.children):
            yield from roots(child, path + (index,))


def phoneset(node):
    offset = 0
    def word():
        nonlocal offset
        value, = struct.unpack_from("<I", node.payload, offset)
        offset += 4
        return value
    def string():
        nonlocal offset
        length = word()
        value = node.payload[offset:offset + length].decode("utf-8")
        offset += length
        return value
    name = string()
    lists = [[string() for _ in range(word())] for _ in range(4)]
    if offset != len(node.payload):
        raise ValueError("Invalid phone table")
    return name, lists


def normalize(node, x, inverse=False):
    values = [torch.from_numpy(tensor(c)) for c in node.children]
    if node.type == "cmpu1":
        mean, scale = values
        return x * scale + mean if inverse else (x - mean) / scale.clamp_min(1e-5)
    lower, upper, target = values
    width = target[1] - target[0]
    return lower + (upper - lower) * (x - target[0]) / width if inverse else target[0] + width * (x - lower) / (upper - lower).clamp_min(1e-5)


class Component(nn.Module):
    def __init__(self, student, teacher, role):
        super().__init__()
        self.student, self.teacher, self.role = student, teacher, role
        self.paths = {str(n.index): path for path, n in roots(teacher.root)}
        self.networks = nn.ModuleDict({key: Network(at(student, path)) for key, path in self.paths.items()})
        self.embeddings = nn.ParameterDict()
        self.embedding_nodes = {}
        self.teacher_embeddings = {}
        self.maps = {}
        if role == "duration":
            self.add_embedding("phone", (3, 0), self.labels(student, (0, 1), 0), self.labels(teacher, (0, 1), 0))
            self.add_embedding("language", (2, 0), self.languages(student, (0, 0)), self.languages(teacher, (0, 0)))
            self.add_embedding("position", (4, 0), list(range(8)), list(range(8)))
            self.add_embedding("speaker", (5,))
        elif role == "pitch":
            self.add_embedding("phone", (0, 3, 0, 0), self.labels(student, (0, 0, 1), 3), self.labels(teacher, (0, 0, 1), 3))
            self.add_embedding("language", (0, 3, 1, 0), self.languages(student, (0, 0, 0)), self.languages(teacher, (0, 0, 0)))
            self.add_embedding("speaker", (0, 1))
        for module in self.modules():
            if isinstance(module, Network) and module.kind == "_ncwnv0":
                module.parts[module.stages * 3 - 1].requires_grad_(False)
        matrices = [n for module in self.modules() if isinstance(module, Network) for n in module.tensor_nodes]
        student.make_dense(matrices + [n for n in self.embedding_nodes.values() if n.type != "prim1"])

    @staticmethod
    def labels(model, path, index):
        return phoneset(at(model, path))[1][index]

    @staticmethod
    def languages(model, path):
        return [phoneset(n)[0] for n in at(model, path).children]

    def add_embedding(self, key, path, source_labels=None, target_labels=None):
        source, target = at(self.student, path), at(self.teacher, path)
        self.embedding_nodes[key] = source
        self.embeddings[key] = nn.Parameter(torch.from_numpy(tensor(source)))
        self.teacher_embeddings[key] = torch.from_numpy(tensor(target))
        if source_labels is not None:
            if (len(set(source_labels)) != len(source_labels) or len(set(target_labels)) != len(target_labels)
                    or len(source_labels) != self.embeddings[key].shape[1]
                    or len(target_labels) != self.teacher_embeddings[key].shape[1]):
                raise ValueError(f"Invalid or ambiguous {key} labels")
            self.maps[key] = [source_labels.index(label) if label in source_labels else -1 for label in target_labels]

    def lookup(self, key, observed):
        table = self.teacher_embeddings[key].T
        distance = (observed[:, None, :] - table[None, :, :]).abs().amax(dim=2)
        error, indices = distance.min(dim=1)
        if bool((error > 1e-5).any()):
            raise ValueError(f"Cannot identify teacher {key} embedding")
        mapped = [self.maps[key][i] for i in indices.tolist()]
        if min(mapped) < 0:
            raise ValueError(f"Teacher {key} category absent from student")
        return self.embeddings[key][:, mapped].T

    def inputs(self, key, x):
        path = self.paths[key]
        if self.role == "duration":
            # duration 输入按语义重建，避免直接使用老师的嵌入下标。
            front_s, front_t = self.student.root.children[0], self.teacher.root.children[0]
            duration = torch.atan2(x[:, 96:97], x[:, 112:113])
            duration = normalize(front_s.children[2], normalize(front_t.children[2], duration, True))
            def encode(values, width):
                angles = values * torch.exp2(torch.arange(width // 2) * (-26.575424 / width))
                return torch.cat((angles.sin(), angles.cos()), dim=1)
            if not torch.allclose(encode(torch.atan2(x[:, 96:97], x[:, 112:113]), 32), x[:, 96:128], atol=1e-5, rtol=1e-5):
                raise ValueError("Ambiguous duration encoding in trace")
            if not torch.allclose(encode(torch.atan2(x[:, 128:129], x[:, 136:137]), 16), x[:, 128:144], atol=1e-5, rtol=1e-5):
                raise ValueError("Ambiguous pitch encoding in duration trace")
            lo_t, hi_t = struct.unpack("<ii", front_t.payload)
            lo_s, hi_s = struct.unpack("<ii", front_s.payload)
            pitch = (torch.atan2(x[:, 128:129], x[:, 136:137]) * (hi_t - lo_t) + lo_t - lo_s) / (hi_s - lo_s)
            return torch.cat((self.lookup("phone", x[:, :32]), self.lookup("language", x[:, 32:64]),
                              self.lookup("position", x[:, 64:80]), self.lookup("position", x[:, 80:96]),
                              encode(duration, 32), encode(pitch, 16), self.embeddings["speaker"].expand(len(x), -1)), dim=1)
        if self.role == "pitch":
            if path == (0, 3, 2):
                raw = normalize(at(self.teacher, (0, 0, 2)), x, True)
                return normalize(at(self.student, (0, 0, 2)), raw)
            if path == (0, 3, 3):
                return self.lookup("phone", x)
            if path == (0, 3, 4):
                # 老师上下文中包含增强后的 speaker，保留增强量。
                speaker = x[:, 264:296] - self.teacher_embeddings["speaker"] + self.embeddings["speaker"]
                return torch.cat((x[:, :264], speaker, self.lookup("language", x[:, 296:328])), dim=1)
        return x

    def target(self, y):
        if self.role == "duration":
            return normalize(at(self.student, (0, 3)), normalize(at(self.teacher, (0, 3)), y, True))
        return y

    def export(self):
        for network in self.networks.values():
            network.export()
        for name, value in self.embeddings.items():
            replace_tensor(self.embedding_nodes[name], value.detach().numpy())
        return self.student.to_bytes()


def samples(directory, component):
    cases = []
    tag = format(int.from_bytes(component.teacher.root.tag, "little"), "x")
    teachers = {key: Network(at(component.teacher, path)).requires_grad_(False) for key, path in component.paths.items()}
    seen = set()
    max_error = 0.0
    for path in sorted(directory.glob("*/sample.json")):
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        if row["model_type"] != tag:
            continue
        key = str(row["node"])
        if key not in teachers:
            raise ValueError(f"Unexpected trace network {key}")
        def read(name):
            value = np.fromfile(path.parent / (name + ".f32"), dtype="<f4")
            result = torch.from_numpy(value.reshape(row[name + "_frames"], row[name + "_channels"]))
            if not bool(torch.isfinite(result).all()):
                raise ValueError("Non-finite trace")
            return result
        x, output = read("input"), read("output")
        condition = read("condition") if "condition_frames" in row else None
        with torch.no_grad():
            expected = teachers[key](x, condition)
            error = float((expected - output).abs().max())
            if not torch.allclose(expected, output, rtol=2e-4, atol=2e-4):
                raise ValueError(f"Teacher parity failed: {key}, max_error={error}")
            target = component.target(expected)
        cases.append((key, x, condition, target))
        seen.add(key)
        max_error = max(max_error, error)
    if seen != set(component.paths):
        raise ValueError(f"Missing network traces: {set(component.paths) - seen}")
    emit(event="trace", role=component.role, samples=len(cases), max_cpp_error=max_error)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acoustic-model", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or not 0 < args.lr < 1 or not 0 < args.seconds < float("inf"):
        parser.error("Invalid training limits")
    torch.set_num_threads(args.threads)
    if args.resume and not args.output.is_dir():
        parser.error("Resume output directory does not exist")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    acoustic = args.acoustic_model or args.voice / "base_model.dnni"
    Model.read(acoustic)
    trace_hash = hashlib.sha256()
    for file in sorted(args.traces.glob("*/*/*")):
        if file.is_file():
            trace_hash.update(str(file.relative_to(args.traces)).encode())
            trace_hash.update(bytes.fromhex(digest(file)))
    started_run = time.monotonic()
    lock = args.output / "run.lock"
    with lock.open("x"):
        pass
    try:
        execute(args, acoustic, trace_hash.hexdigest(), started_run)
    finally:
        lock.unlink()


def execute(args, acoustic, trace_hash, started_run):
    report = {"method": "module_distillation", "roles": {}}
    for role, filename in FILES.items():
        started = time.monotonic()
        student, teacher = Model.read(args.voice / filename), Model.read(args.teacher / filename)
        component = Component(student, teacher, role)
        train, validation = [samples(args.traces / split, component) for split in ("train", "validation")]
        parameters = [p for p in component.parameters() if p.requires_grad]
        initial = {name: p.detach().clone() for name, p in component.named_parameters()}
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        contract = dict(source=digest(args.voice / filename), teacher=digest(args.teacher / filename),
                        acoustic=digest(acoustic), traces=trace_hash, lr=args.lr, threads=args.threads, version=1)
        def loss(case):
            key, x, condition, target = case
            return (component.networks[key](component.inputs(key, x), condition) - target).square().mean() / target.square().mean().clamp_min(1e-4)
        @torch.no_grad()
        def evaluate():
            return sum(float(loss(case)) for case in validation) / len(validation)
        baseline = best_loss = evaluate()
        best = {k: v.clone() for k, v in component.state_dict().items()}
        gradients = set()
        step = 0
        checkpoint = args.output / (role + '.pt')
        if args.resume and checkpoint.exists():
            saved = torch.load(checkpoint, weights_only=True)
            if saved['contract'] != contract:
                raise ValueError("Resume model, traces or settings changed")
            component.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            step, best_loss, best = saved['step'], saved['best_loss'], saved['best']
            gradients = set(saved['gradients'])
        def save():
            atomic_save(dict(step=step, model=component.state_dict(), optimizer=optimizer.state_dict(), best=best,
                             best_loss=best_loss, contract=contract, gradients=sorted(gradients)), checkpoint)
        save()
        while step < args.steps and time.monotonic() - started_run < args.seconds:
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for case in train:
                value = loss(case) / len(train)
                if not bool(torch.isfinite(value)):
                    raise ValueError("Non-finite training loss")
                value.backward()
                total += float(value.detach())
            missing = [name for name, p in component.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise ValueError(f"Disconnected parameters: {missing}")
            gradients.update(name for name, p in component.named_parameters() if p.requires_grad and bool(torch.count_nonzero(p.grad)))
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            score = evaluate()
            if not np.isfinite(score):
                raise ValueError("Non-finite validation loss")
            if score <= best_loss:
                best_loss, best = score, {k: v.clone() for k, v in component.state_dict().items()}
            step += 1
            save()
            emit(event="component_step", role=role, step=step, loss=total, validation=score)
        component.load_state_dict(best)
        changed = sum(not torch.equal(initial[name], p) for name, p in component.named_parameters())
        output = component.export()
        Model(output)
        (args.output / filename).write_bytes(output)
        report["roles"][role] = dict(parameters=sum(p.numel() for p in parameters), gradient_tensors=len(gradients),
                                    trainable_tensors=len(parameters),
                                    inactive_parameters=sum(p.numel() for p in component.parameters() if not p.requires_grad),
                                    changed_tensors=changed, baseline=baseline, validation=best_loss,
                                    step=step, complete=step >= args.steps,
                                    seconds=time.monotonic() - started)
        emit(event="component_complete", role=role, **report["roles"][role])
    shutil.copyfile(acoustic, args.output / "base_model.dnni")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        emit(event="error", message=str(error))
        raise SystemExit(1)

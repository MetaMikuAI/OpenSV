import argparse
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .format import Model
from .network import AcousticCore, Network
from .runtime import atomic_save, digest, emit


def load_data(path, core, scope="embeddings", objective=None):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported dataset version")
    if manifest.get("supervision", "features") not in ("features", "audio"):
        raise ValueError("Unsupported supervision")
    if (manifest.get("supervision", "features") == "audio") != (objective is not None):
        raise ValueError("Audio datasets require --vocoder; feature datasets must omit it")
    groups = {"train": [], "validation": []}
    signatures, ids, files, audio_files = [], set(), set(), set()
    for row in manifest["samples"]:
        name, split = row["id"], row["split"]
        source = (path.parent / row["features"]).resolve()
        signature = digest(source)
        if not isinstance(name, str) or not name or name in ids or signature in files:
            raise ValueError("Duplicate sample ID or feature content")
        if split not in groups:
            raise ValueError("Expected train or validation split")
        ids.add(name)
        files.add(signature)
        signatures.append((name, split, signature))
        with np.load(source, allow_pickle=False) as archive:
            def matrix(key, width):
                value = np.asarray(archive[key], dtype=np.float32)
                if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
                    raise ValueError(f"{name}: invalid {key} shape or values")
                return torch.from_numpy(value.copy())
            phones, pitch = matrix("phones", 108), matrix("pitch", 1)
            noise = matrix("noise", 4)
            if objective is None:
                target = matrix("target", 71)
            else:
                from .audio import read_wav
                audio_path = (path.parent / row["audio"]).resolve()
                waveform = read_wav(audio_path, objective.vocoder.sample_rate)
                if len(waveform) != len(pitch) * objective.vocoder.hop:
                    raise ValueError(f"{name}: WAV length does not match aligned frames")
                log_f0 = matrix("log_f0", 1)
                if (len(log_f0) != len(pitch) or not bool(log_f0.exp().isfinite().all())
                        or not bool((log_f0.exp() > 0).all())):
                    raise ValueError("Invalid aligned F0")
                target = (log_f0, waveform)
                audio_signature = digest(audio_path)
                if audio_signature in audio_files:
                    raise ValueError("Duplicate target WAV content")
                audio_files.add(audio_signature)
                signatures.append((name, "audio", audio_signature))
            counts, languages = archive["counts"], archive["languages"]
            for value in (counts, languages):
                if value.shape != (len(phones),) or value.dtype.kind not in "iu":
                    raise ValueError(f"{name}: invalid phoneme counts/languages")
            if len(noise) != len(pitch) or (objective is None and len(target) != len(pitch)):
                raise ValueError(f"{name}: frame counts differ")
            with torch.no_grad():
                context = core.frame_context(phones, counts.tolist(), languages.tolist(), pitch)
        inputs = context if scope == "embeddings" else (phones, counts.tolist(), languages.tolist(), pitch)
        groups[split].append((inputs, noise, target))
    if not all(groups.values()):
        raise ValueError("Dataset requires separate train and validation samples")
    return groups, signatures


def run(args):
    if args.lr is None:
        args.lr = 0.000001 if args.vocoder is not None else 0.00001 if args.scope == "acoustic" else 0.001
    if args.steps < 1 or args.seconds <= 0 or not math.isfinite(args.seconds):
        raise ValueError("Steps and seconds must be positive")
    if args.threads < 1 or args.save_every < 1 or not 0 < args.lr < 1:
        raise ValueError("Invalid threads, save interval or learning rate")
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = Model.read(args.model)
    core = AcousticCore(model.root)
    expected_model = json.loads(args.dataset.read_text(encoding="utf-8")).get("model_sha256")
    if expected_model is not None and expected_model != digest(args.model):
        raise ValueError("Dataset was captured for a different base model")
    core.requires_grad_(args.scope == "acoustic")
    parameters = core.training_parameters() if args.scope == "acoustic" else [core.speaker, core.style]
    for parameter in parameters:
        parameter.requires_grad_(True)
    if args.scope == "acoustic":
        matrices = [node for module in core.modules() if isinstance(module, Network) for node in module.tensor_nodes]
        model.make_dense([*matrices, core.language_node])
    objective = None
    if args.vocoder is not None:
        from .audio import AudioObjective
        if args.scope != "acoustic":
            raise ValueError("Audio supervision requires acoustic scope")
        objective = AudioObjective(model, args.vocoder)
    groups, signatures = load_data(args.dataset, core, args.scope, objective)
    contract = {"model": digest(args.model), "samples": signatures,
                "scope": args.scope, "lr": args.lr, "seed": args.seed,
                "threads": args.threads, "backend_version": 2 if args.scope == "acoustic" else 1}
    if objective is not None:
        contract.update(vocoder=objective.signature, loss="multiresolution_stft_v1")
    output = args.output
    state = None
    if args.resume:
        if not output.is_dir():
            raise ValueError("Resume output directory does not exist")
    elif output.exists():
        raise ValueError("Output directory already exists; use --resume or a new directory")
    else:
        output.mkdir(parents=True)
    lock = output / "run.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if args.resume:
            state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            if state["contract"] != contract:
                raise ValueError("Resume requires unchanged model, dataset and training settings")
        return train(args, model, core, parameters, groups, contract, state, started, rng, objective)
    finally:
        lock.unlink()


def train(args, model, core, parameters, groups, contract, state, started, rng, objective=None):
    output = args.output
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    step, best_loss = 0, float("inf")
    cursor, order = 0, []

    def vectors(values):
        if len(values) != len(parameters):
            raise ValueError("Checkpoint parameter count mismatch")
        with torch.no_grad():
            for parameter, value in zip(parameters, values):
                if parameter.shape != value.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid checkpoint parameter")
                parameter.copy_(value)

    if state is not None:
        vectors(state["vectors"])
        optimizer.load_state_dict(state["optimizer"])
        step, best_loss = state["step"], state["best_loss"]
        cursor, order = state["cursor"], state["order"]
        rng.setstate(state["rng"])

    def forward(item):
        context, noise, target = item
        if args.scope == "acoustic":
            inputs = core.frame_context(*context)
        else:
            frames = len(context)
            inputs = torch.cat((context[:, :72], core.speaker.expand(frames, -1),
                                context[:, 104:112], core.style.expand(frames, -1)), dim=1)
        prediction = core(inputs, noise)
        return objective(prediction, target) if objective is not None else (prediction - target).square().mean()

    @torch.no_grad()
    def evaluate():
        score = sum(float(forward(item)) for item in groups["validation"]) / len(groups["validation"])
        if not math.isfinite(score):
            raise ValueError("Non-finite validation loss")
        return score

    def save():
        nonlocal best_loss, best_vectors, best_step
        score = evaluate()
        # 同一检查点保存最新值与最优值，避免跨文件续训不一致。
        if score <= best_loss:
            best_loss = score
            best_vectors = [p.detach().clone() for p in parameters]
            best_step = step
        atomic_save({"contract": contract, "step": step, "vectors": [p.detach().clone() for p in parameters],
                     "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                     "best_vectors": best_vectors, "best_step": best_step,
                     "gradient_names": sorted(gradient_names),
                     "cursor": cursor, "order": order, "rng": rng.getstate()}, output / "latest.pt")
        emit(event="checkpoint", step=step, validation_loss=score, loss_kind=contract.get("loss", "feature_mse"), best_step=best_step)

    best_vectors = state["best_vectors"] if state else [p.detach().clone() for p in parameters]
    best_step = state["best_step"] if state else 0
    gradient_names = set(state.get("gradient_names", [])) if state else set()
    parameter_count = sum(p.numel() for p in parameters)
    emit(event="ready", backend="pytorch", scope=args.scope, trained_parameters=parameter_count,
         inactive_parameters=sum(p.numel() for p in core.parameters() if not p.requires_grad),
         train_samples=len(groups["train"]), validation_samples=len(groups["validation"]), step=step)
    if state is None:
        save()
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        while not interrupted and step < args.steps and time.monotonic() - started < args.seconds:
            if cursor >= len(order):
                order = list(range(len(groups["train"])))
                rng.shuffle(order)
                cursor = 0
            item = groups["train"][order[cursor]]
            optimizer.zero_grad(set_to_none=True)
            loss = forward(item)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite training loss")
            loss.backward()
            missing = [name for name, p in core.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise ValueError(f"Disconnected trainable parameters: {missing}")
            gradient_names.update(name for name, p in core.named_parameters()
                                  if p.requires_grad and bool(torch.count_nonzero(p.grad)))
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor += 1
            step += 1
            emit(event="step", step=step, loss=float(loss.detach()))
            if step % args.save_every == 0:
                save()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    save()
    vectors(best_vectors)
    core.export()
    destination = output / "trained-base.dnni"
    temporary = output / "trained-base.tmp"
    temporary.write_bytes(model.to_bytes())
    Model.read(temporary)
    temporary.replace(destination)
    report = {"backend": "pytorch", "scope": args.scope, "trained_parameters": parameter_count,
              "parameter_tensors": len(parameters), "gradient_tensors": len(gradient_names),
              "inactive_parameters": sum(p.numel() for p in core.parameters() if not p.requires_grad),
              "step": step, "best_step": best_step, "validation_loss": best_loss,
              "loss_kind": contract.get("loss", "feature_mse"),
              "complete": step >= args.steps, "interrupted": interrupted,
              "model": str(destination.resolve()), "seconds": time.monotonic() - started}
    if objective is None:
        report["validation_mse"] = best_loss
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    emit(event="complete", **report)
    return 130 if interrupted else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scope", choices=("embeddings", "acoustic"), default="acoustic")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vocoder", type=Path)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        emit(event="error", message=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

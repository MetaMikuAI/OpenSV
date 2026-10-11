import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import os
import shutil

import numpy as np
import parselmouth
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch

from .format import Model
from .runtime import digest, emit
from .score import load_score
from .vocoder import Normalizer


def read_audio(path):
    rate, data = wavfile.read(path)
    if not 8000 <= rate <= 192000 or data.ndim not in (1, 2) or len(data) / rate > 660:
        raise ValueError("WAV 采样率、通道或长度不支持（最长 11 分钟）")
    if data.dtype.kind == "i":
        audio = data.astype(np.float64) / 2 ** (data.dtype.itemsize * 8 - 1)
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float64) - 128) / 128
    elif data.dtype.kind == "f":
        audio = data.astype(np.float64)
    else:
        raise ValueError("WAV 编码不支持")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("WAV 为空或包含无效采样")
    warnings = []
    if np.max(np.abs(audio)) >= 0.999:
        warnings.append("音频接近或超过满幅，请检查削波。")
    if audio.ndim == 2:
        if audio.shape[1] != 2:
            raise ValueError("只支持单声道或双声道 WAV")
        if np.sqrt(np.mean((audio[:, 0] - audio[:, 1]) ** 2)) > 0.001:
            warnings.append("左右声道不同，已取平均；请确认没有伴奏、混响或叠唱。")
        audio = audio.mean(axis=1)
    if np.sqrt(np.mean(audio ** 2)) < 1e-5:
        raise ValueError("音频基本静音，无法训练")
    return rate, audio.astype(np.float32), warnings


def alignment(audio, rate, intervals, offset):
    hop = round(rate * .02)
    count = len(audio) // hop
    energy = np.sqrt(np.mean(audio[:count * hop].reshape(count, hop) ** 2, axis=1))
    active = energy > max(1e-4, float(energy.max()) * .04)
    times = (np.arange(count) + .5) * hop / rate
    expected = np.zeros(count, dtype=bool)
    for start, end, _ in intervals:
        expected |= (times >= start + offset - .08) & (times <= end + offset + .08)
    coverage = np.count_nonzero(active & expected) / max(1, np.count_nonzero(active))
    recall = np.count_nonzero(active & expected) / max(1, np.count_nonzero(expected))
    warnings, errors = [], []
    if intervals[0][0] + offset < -.05 or intervals[-1][1] + offset > len(audio) / rate + .05:
        errors.append("工程音符超出 WAV 时间范围，请检查导出起点、尾部或音频偏移。")
    if coverage < .75:
        warnings.append(f"仅 {coverage:.0%} 的有声区位于预计音符附近，可能存在整体错位、伴奏或多余片段。")
    if recall < .55:
        warnings.append(f"预计音符区只有 {recall:.0%} 检测到声音，请检查缺失片段或时间错位。")
    if coverage < .3 or recall < .2:
        errors.append("音符与有声区明显不匹配，修正后再导入。")
    return warnings, errors, dict(audio_coverage=coverage, note_coverage=recall)


def crop(audio, first, length):
    result = np.zeros(length, np.float32)
    start, end = max(first, 0), min(first + length, len(audio))
    if end > start:
        result[start - first:end - first] = audio[start:end]
    return result


def prepare(args):
    if not math.isfinite(args.offset) or abs(args.offset) > 600:
        raise ValueError("音频偏移无效")
    project, intervals = load_score(args.project)
    if not args.dictionary and any(not note.get("phonemes", "").strip()
                                   for track in project["tracks"] for note in track["mainGroup"]["notes"]):
        raise ValueError("工程只有歌词，缺少发音词典。请展开「引擎与发音词典」并填写词典目录或 JSON 文件，再检查切分。")
    rate, audio, warnings = read_audio(args.audio)
    checks, errors, metrics = alignment(audio, rate, intervals, args.offset)
    warnings += checks
    warnings.append("音素边界由底模根据歌词与音符估计，尚未进行逐音素强制对齐；请试听切片。")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    report = dict(warnings=warnings, errors=errors, metrics=metrics, segments=[],
                  audio_sha256=digest(args.audio), project_sha256=digest(args.project), offset=args.offset)
    def save_report():
        (root / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(event="prepared", **report)
    if errors:
        save_report()
        return
    normalized = root / "project.svp"
    normalized.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    command = [str(args.engine), "dataset", "condition", "--voice", str(args.voice),
               "--project", str(normalized), "--output", str(root), "--language", args.language]
    if args.dictionary:
        command += ["--dictionary", str(args.dictionary)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"引擎准备失败，退出码 {result.returncode}")
    conditions = json.loads((root / "conditioning.json").read_text(encoding="utf-8"))
    target_rate = int(conditions["sample_rate"])
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, rate // divisor).astype(np.float32)
    normalizer = Normalizer(next(n for n in Model.read(root / "base_model.dnni").nodes
                                if n.type in ("_ftmfv2", "_ftmfv3")).children[4])
    hop = round(target_rate * .005)
    samples, signatures = [], set()
    for index, condition in enumerate(conditions["phrases"]):
        counts = np.asarray(condition["counts"], np.int64)
        frames = int(counts.sum())
        start = condition["start"] + args.offset
        waveform = crop(audio, round(start * target_rate), frames * hop)
        if np.sqrt(np.mean(waveform ** 2)) < 1e-4:
            warnings.append(f"片段 {index + 1} 基本静音，已跳过。")
            continue
        signature = hashlib.sha256(waveform.tobytes()).hexdigest()
        if signature in signatures:
            warnings.append(f"片段 {index + 1} 音频重复，已跳过。")
            continue
        signatures.add(signature)
        measured = parselmouth.Sound(waveform, target_rate).to_pitch_ac(
            time_step=.005, pitch_floor=50, pitch_ceiling=1600)
        f0 = measured.selected_array["frequency"]
        voiced = f0 > 0
        if np.count_nonzero(voiced) < 3:
            warnings.append(f"片段 {index + 1} 无法提取可靠音高，已跳过。")
            continue
        times = np.arange(frames) * .005
        log_f0 = np.interp(times, measured.xs()[voiced], np.log(f0[voiced])).astype(np.float32).reshape(-1, 1)
        difference = np.abs(np.interp(measured.xs()[voiced], times, condition["log_f0"]) - np.log(f0[voiced])) * 12 / np.log(2)
        if np.median(difference) > 2:
            warnings.append(f"片段 {index + 1} 实测音高与音符偏差较大，请检查移调、错位或音高提取。")
        name = f"segment-{index + 1:04d}"
        np.savez(root / f"{name}.npz", phones=np.asarray(condition["phones"], np.float32).reshape(-1, 108),
                 counts=counts, languages=np.asarray(condition["languages"], np.int64),
                 noise=np.asarray(condition["noise"], np.float32).reshape(frames, 4),
                 pitch=normalizer(torch.from_numpy(log_f0)).detach().numpy(), log_f0=log_f0)
        wavfile.write(root / f"{name}.wav", target_rate, waveform)
        samples.append(dict(id=name, features=f"{name}.npz", audio=f"{name}.wav"))
        report["segments"].append(dict(id=name, start=start, end=start + frames * .005))
        emit(event="segment", count=len(samples))
    split = max(1, len(samples) - max(1, round(len(samples) * .2)))
    if len(samples) > split:
        boundary = report["segments"][split]["start"]
        keep = [i for i in range(split) if report["segments"][i]["end"] <= boundary]
        if len(keep) != split and not getattr(args, "defer_split", False):
            warnings.append("训练与验证交界处的重叠片段已跳过，避免重复音频。")
    else:
        keep = []
    selected = []
    for index in keep + list(range(split, len(samples))):
        part = "train" if index < split else "validation"
        selected.append(dict(samples[index], split=part))
        report["segments"][index]["split"] = part
    if getattr(args, "defer_split", False):
        keep = list(range(len(samples)))
        split = len(samples)
        selected = [dict(sample, split="train") for sample in samples]
        for segment in report["segments"]:
            segment["split"] = "train"
    if not samples or (not getattr(args, "defer_split", False) and (not keep or len(samples) <= split)):
        errors.append("独立片段不足，至少需要两个不重叠的有效片段；请提供更长或含停顿的材料。")
    report.update(train_samples=len(keep), validation_samples=len(samples) - split if len(samples) > split else 0,
                  seconds=sum(s["end"] - s["start"] for s in report["segments"] if "split" in s))
    if not errors:
        manifest = dict(version=1, supervision="audio", model_sha256=digest(root / "base_model.dnni"), samples=selected)
        (root / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_report()


def prepare_batch(args, pairs):
    if not 1 <= len(pairs) <= 100:
        raise ValueError("请提供 1–100 组对应材料")
    if len(pairs) == 1:
        return prepare(argparse.Namespace(**{**vars(args), **pairs[0]}))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    report = dict(warnings=[], errors=[], segments=[], train_samples=0, validation_samples=0, seconds=0)
    samples, seen, sources = [], set(), set()
    boundary = len(pairs) - max(1, round(len(pairs) * .2))
    for index, pair in enumerate(pairs):
        name = f"source-{index + 1:03d}"
        destination = root / name
        emit(event="source", current=index + 1, total=len(pairs))
        prepare(argparse.Namespace(**{**vars(args), **pair, "output": destination, "defer_split": True}))
        part = json.loads((destination / "preparation.json").read_text(encoding="utf-8"))
        if part["audio_sha256"] in sources:
            part["errors"].append("重复的源 WAV，请移除重复材料。")
        sources.add(part["audio_sha256"])
        for key in ("warnings", "errors"):
            report[key].extend(f"材料 {index + 1}：{message}" for message in part[key])
        if part["errors"]:
            continue
        manifest = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
        split = "train" if index < boundary else "validation"
        for sample, segment in zip(manifest["samples"], [s for s in part["segments"] if s.get("split")]):
            signature = digest(destination / sample["audio"])
            if signature in seen:
                report["errors"].append(f"材料 {index + 1} 含重复音频，请移除重复材料后重新导入。")
                continue
            seen.add(signature)
            samples.append(dict(sample, id=name + "/" + sample["id"], split=split,
                                audio=name + "/" + sample["audio"], features=name + "/" + sample["features"]))
            report["segments"].append(dict(segment, id=name + "/" + segment["id"], split=split, source=index + 1))
            report[split + "_samples"] += 1
            report["seconds"] += segment["end"] - segment["start"]
    report["warnings"].append("多组材料按来源划分训练与验证，最后约 20% 的材料用于验证。")
    if not report["train_samples"] or not report["validation_samples"]:
        report["errors"].append("训练与验证都需要有效材料。")
    if not report["errors"]:
        for filename in ("base_model.dnni", "pitch_model.dnni"):
            shutil.copyfile(root / "source-001" / filename, root / filename)
        manifest = dict(version=1, supervision="audio", model_sha256=digest(root / "base_model.dnni"), samples=samples)
        (root / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(event="prepared", **report)


def main():
    parser = argparse.ArgumentParser(description="从对应的 WAV 与 SVP／VSQX 准备训练数据")
    for name in ("voice", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--pairs", type=Path, help="含 pairs 数组的 JSON，每组含 project、audio、offset")
    parser.add_argument("--engine", default="opensv")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--language", default="mandarin")
    parser.add_argument("--offset", type=float, default=0, help="WAV 时间减去工程时间，单位秒")
    args = parser.parse_args()
    try:
        if args.pairs:
            pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"]
            prepare_batch(args, pairs)
        elif args.project and args.audio:
            prepare(args)
        else:
            raise ValueError("请提供对应的工程与音频，或 --pairs 清单")
        report = json.loads((args.output / "preparation.json").read_text(encoding="utf-8"))
        return 2 if report["errors"] else 0
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        emit(event="error", message=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

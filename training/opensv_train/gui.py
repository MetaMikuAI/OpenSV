import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
from urllib.parse import parse_qs, urlsplit


def configuration(values):
    result = {key: str(values.get(key, "")).strip() for key in (
        "engine", "voice", "project", "audio", "mode",
        "dataset", "output", "dictionary", "preview", "preview_language", "preview_dictionary", "language")}
    result["mode"] = result["mode"] or ("dataset" if result["dataset"] else "pair")
    if result["mode"] not in ("pair", "dataset"):
        raise ValueError("数据来源无效")
    result["offset"] = float(values.get("offset", 0))
    if not math.isfinite(result["offset"]) or abs(result["offset"]) > 600:
        raise ValueError("音频偏移必须在 -600 到 600 秒之间")
    result["language"] = result["language"] or "mandarin"
    if result["preview_language"] not in ("", "mandarin", "japanese"):
        raise ValueError("试听语言必须为中文或日语，或跟随材料语言")
    result["device"] = str(values.get("device", "cpu")).strip().lower() or "cpu"
    if result["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("训练设备必须为 cpu、cuda 或 mps")
    pairs = values.get("pairs", [dict(project=result["project"], audio=result["audio"], offset=result["offset"])])
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 100:
        raise ValueError("请提供 1–100 组材料")
    result["pairs"] = []
    for pair in pairs:
        offset = float(pair.get("offset", 0))
        if not math.isfinite(offset) or abs(offset) > 600:
            raise ValueError("材料偏移必须在 -600 到 600 秒之间")
        result["pairs"].append(dict(project=str(Path(pair.get("project", "")).resolve()) if pair.get("project") else "",
                                   audio=str(Path(pair.get("audio", "")).resolve()) if pair.get("audio") else "", offset=offset))
    if not result["output"]:
        raise ValueError("请选择工作目录")
    for key, default, low, high in (("steps", 100, 1, 10000000),
                                   ("seconds", 1800, 1, 604800),
                                   ("threads", 4, 1, 256)):
        result[key] = int(values.get(key, default))
        if not low <= result[key] <= high:
            raise ValueError(f"{key} 必须在 {low}–{high} 之间")
    epochs = values.get("epochs")
    result["epochs"] = None if epochs is None or str(epochs).strip() == "" else float(epochs)
    if result["epochs"] is not None and (not math.isfinite(result["epochs"]) or result["epochs"] <= 0):
        raise ValueError("训练轮数必须为大于 0 的有限数值")
    result["lr"] = float(values.get("lr", 0.000001))
    if not math.isfinite(result["lr"]) or not 0 < result["lr"] <= 1:
        raise ValueError("学习率必须大于 0 且不超过 1")
    return result


def training_target(config, manifest):
    if manifest.get("supervision") != "audio":
        raise ValueError("界面仅接受 audio 干声数据集")
    count = sum(sample.get("split") == "train" for sample in manifest["samples"])
    if not count:
        raise ValueError("数据集中没有训练片段")
    target = config["steps"] if config["epochs"] is None else config["epochs"] * count
    if not math.isfinite(target) or target > 10000000:
        raise ValueError("训练目标不能超过 10000000 步，请减少训练轮数")
    steps = math.ceil(target)
    return dict(config, steps=steps, train_samples=count, total_epochs=steps / count)


def prepared_target(config, dataset):
    if not dataset.is_file():
        raise ValueError("缺少数据集清单，请先完成检查并切分")
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not isinstance(manifest.get("samples"), list)
            or any(not isinstance(sample, dict) for sample in manifest["samples"])):
        raise ValueError("数据集清单格式无效")
    config = training_target(config, manifest)
    if not any(sample.get("split") == "validation" for sample in manifest["samples"]):
        raise ValueError("训练与验证都需要有效材料。")
    paths = [dataset.parent / name for name in ("base_model.dnni", "pitch_model.dnni")]
    for sample in manifest["samples"]:
        for key in ("features", "audio"):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError("数据集清单格式无效")
            paths.append(dataset.parent / sample[key])
    for path in paths:
        if not path.is_file():
            raise ValueError(f"数据集文件不存在：{path}")
    return config


def device_options():
    available, version, reason = False, None, ""
    mps_available, mps_reason = False, ""
    try:
        import torch
        version = torch.version.cuda
        available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        if not mps_available:
            mps_reason = "MPS 不可用，请使用支持 Metal 的 Mac 与支持 MPS 的 PyTorch，或选择 CPU"
        if not available:
            reason = ("未检测到可用的 NVIDIA GPU 或驱动" if version else
                      "当前 PyTorch 不支持 CUDA；NVIDIA GPU 需要支持 CUDA 的 PyTorch")
    except (ImportError, OSError, RuntimeError):
        reason = mps_reason = "PyTorch 加载失败，请检查训练环境"
    return dict(device="cpu", devices=["cpu"] + (["cuda"] if available else []) + (["mps"] if mps_available else []),
                cuda_available=available, cuda_version=version, cuda_reason=reason,
                mps_available=mps_available, mps_reason=mps_reason)


class Jobs:
    def __init__(self):
        self.lock = threading.RLock()
        self.worker = None
        self.stop_file = None
        self.losses = deque(maxlen=20)
        self.state = {"busy": False, "phase": "idle", "logs": [], "artifacts": []}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def log(self, line):
        with self.lock:
            self.state["logs"].append(line)
            self.state["logs"] = self.state["logs"][-400:]
            try:
                event = json.loads(line)
                if isinstance(event, dict) and "event" in event:
                    if event["event"] == "step" and isinstance(event.get("loss"), (int, float)) and math.isfinite(event["loss"]):
                        self.losses.append(event["loss"])
                        event["loss_average"] = sum(self.losses) / len(self.losses)
                    self.state["event"] = event
                    metrics = self.state.setdefault("metrics", {})
                    metrics.update({k: event[k] for k in ("step", "loss", "loss_average", "validation_loss", "best_step", "train_samples") if k in event})
                    if event["event"] in ("step", "checkpoint"):
                        history = self.state.setdefault("history", [])
                        history.append({k: event[k] for k in ("step", "loss", "loss_average", "validation_loss") if k in event})
                        if len(history) > 2000:
                            self.state["history"] = history[::2]
                    return event
            except ValueError:
                pass

    def start(self, values, resume=False, prepare=False):
        with self.lock:
            if self.state["busy"]:
                raise ValueError("已有任务运行中")
            config = configuration(values)
            if prepare and config["mode"] != "pair":
                raise ValueError("检查切分需要干声与对应工程")
            root = Path(config["output"]).expanduser().resolve()
            if resume:
                if not (root / "gui.json").is_file():
                    raise ValueError("工作目录中没有可续训记录，请选择之前训练使用的工作目录")
                saved = json.loads((root / "gui.json").read_text(encoding="utf-8"))
                saved.update(steps=config["steps"], seconds=config["seconds"], epochs=config["epochs"])
                if "device" in values:
                    saved["device"] = config["device"]
                saved.update({key: config[key] for key in ("engine", "preview", "preview_language", "preview_dictionary") if key in values})
                config = configuration(saved)
                if not (root / "run/latest.pt").is_file():
                    raise ValueError("工作目录中没有可续训的 latest.pt")
            elif not prepare and config["mode"] == "pair":
                if not all((root / name).is_file() for name in ("gui.json", "dataset/preparation.json")):
                    raise ValueError("请先点击「检查并切分」，完成后确认对齐，再开始训练")
                saved = json.loads((root / "gui.json").read_text(encoding="utf-8"))
                saved = configuration(saved)
                if (any(Path(config[key]).resolve() != Path(saved[key]).resolve()
                        for key in ("voice", "dictionary"))
                        or any(config[key] != saved[key] for key in ("pairs", "language"))):
                    raise ValueError("材料已修改，请使用新工作目录重新检查切分")
                report = json.loads((root / "dataset/preparation.json").read_text(encoding="utf-8"))
                if report["errors"]:
                    raise ValueError("材料检查未通过，请修正后重新导入")
                if not values.get("confirm_alignment"):
                    raise ValueError("请先查看切分结果，并确认已检查对齐提示")
                saved.update({key: config[key] for key in ("steps", "epochs", "seconds", "threads", "lr", "device")})
                saved.update({key: config[key] for key in ("engine", "preview", "preview_language", "preview_dictionary") if key in values})
                config = configuration(saved)
                if (root / "run").exists():
                    raise ValueError("该目录已有训练记录，请使用续训")
            if config["device"] in ("cuda", "mps"):
                devices = device_options()
                device = config["device"]
                if not devices[device + "_available"]:
                    raise ValueError(device.upper() + " 不可用：" + devices[device + "_reason"])
            engine = shutil.which(config["engine"]) or config["engine"]
            if not Path(engine).is_file():
                raise ValueError("找不到 opensv 程序，请先 cargo build --release")
            config["engine"] = str(Path(engine).resolve())
            required = ["voice"]
            if not resume:
                required += ["dataset"] if config["mode"] == "dataset" else []
                if config["mode"] == "pair":
                    for index, pair in enumerate(config["pairs"]):
                        for key in ("project", "audio"):
                            if not pair[key] or not Path(pair[key]).is_file():
                                raise ValueError(f"材料 {index + 1} 的 {key} 文件不存在")
            required += [key for key in ("dictionary", "preview", "preview_dictionary") if config[key]]
            for key in required:
                if not config[key] or not Path(config[key]).exists():
                    raise ValueError(f"路径不存在：{key}")
                config[key] = str(Path(config[key]).resolve())
            create_root = not resume and (prepare or config["mode"] == "dataset")
            if create_root and root.exists():
                raise ValueError("工作目录已存在，请选择新的工作目录；已有训练记录请使用续训")
            preparation = None
            if not prepare:
                dataset = Path(config["dataset"]) if config["mode"] == "dataset" else root / "dataset/dataset.json"
                config = prepared_target(config, dataset)
                report_path = root / "dataset/preparation.json"
                if config["mode"] == "pair" and report_path.is_file():
                    preparation = json.loads(report_path.read_text(encoding="utf-8"))
            if create_root:
                try:
                    root.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    raise ValueError("工作目录已存在，请选择新的工作目录；已有训练记录请使用续训") from None
            config["output"] = str(root)
            (root / "gui.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            self.stop_file = root / "stop.request"
            self.stop_file.unlink(missing_ok=True)
            self.state = {"busy": True, "phase": "starting", "logs": [], "artifacts": [],
                          "output": str(root), "preparation": preparation, "config": config}
            self.losses.clear()
            self.worker = threading.Thread(target=self.run, args=(config, resume, prepare), daemon=False)
            self.worker.start()

    def command(self, args, phase):
        self.update(phase=phase)
        error_message = None
        environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=environment,
                              creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0,
                              start_new_session=os.name != "nt") as process:
            for line in process.stdout:
                event = self.log(line.rstrip())
                if event and event["event"] == "error" and isinstance(event.get("message"), str):
                    error_message = event["message"].strip() or error_message
                elif line.startswith("Error: "):
                    error_message = line.removeprefix("Error: ").strip() or error_message
            code = process.wait()
        if code != 0 and not (phase == "training" and code == 130) and not (phase == "preparing" and code == 2):
            raise RuntimeError(error_message or f"{phase} 失败，退出码 {code}；请查看日志")
        return code

    def run(self, config, resume, prepare=False):
        root = Path(config["output"])
        engine = config["engine"]
        dictionary = ["--dictionary", config["dictionary"]] if config["dictionary"] else []
        try:
            dataset = Path(config["dataset"]) if config["mode"] == "dataset" else root / "dataset/dataset.json"
            if prepare:
                self.command([sys.executable, "-u", "-m", "opensv_train.prepare", "--engine", engine,
                              "--voice", config["voice"], "--pairs", str(root / "gui.json"),
                              "--output", str(dataset.parent), "--language", config["language"], *dictionary], "preparing")
                report = json.loads((dataset.parent / "preparation.json").read_text(encoding="utf-8"))
                selected = [s for s in report["segments"] if s.get("split")]
                audio = str(dataset.parent / (selected[0]["id"] + ".wav")) if selected else None
                self.update(phase="review", preparation=report, audio=audio,
                            artifacts=[str(dataset.parent / "preparation.json")])
                if report["errors"]:
                    return
                manifest = json.loads(dataset.read_text(encoding="utf-8"))
                config = training_target(config, manifest)
                (root / "gui.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                self.update(config=config)
                return
            args = [sys.executable, "-u", "-m", "opensv_train", "--model", str(dataset.parent / "base_model.dnni"),
                    "--vocoder", str(dataset.parent / "pitch_model.dnni"), "--dataset", str(dataset),
                    "--output", str(root / "run"), "--scope", "acoustic", "--stop-file", str(self.stop_file)]
            for key in ("steps", "seconds", "threads", "lr", "device"):
                args += ["--" + key, str(config[key])]
            if resume:
                args.append("--resume")
            code = self.command(args, "training")
            name = "trained-" + time.strftime("%Y%m%d-%H%M%S")
            voice = root / (name + ".nofs")
            self.command([engine, "voice-export", "--voice", config["voice"], "--model",
                          str(root / "run/trained-base.dnni"), "--output", str(voice)], "export")
            self.update(artifacts=[str(voice), str(root / "run/trained-base.dnni")])
            if config["preview"]:
                audio = root / (name + ".wav")
                preview_dictionary = config["preview_dictionary"] or config["dictionary"]
                dictionary = ["--dictionary", preview_dictionary] if preview_dictionary else []
                try:
                    self.command([engine, "render", "--voice", str(voice), "--project", config["preview"],
                                  "--output", str(audio), "--pitch-mode", "notes",
                                  "--language", config["preview_language"] or config["language"], *dictionary], "render")
                    self.update(audio=str(audio))
                except (OSError, RuntimeError) as error:
                    warning = f"声库已导出，试听渲染失败：{error}"
                    self.log(warning)
                    self.update(warning=warning)
            self.update(phase="stopped" if code == 130 else "complete")
        except Exception as error:
            self.log(str(error))
            self.update(phase="error", error=str(error))
        finally:
            self.update(busy=False)

    def stop(self):
        with self.lock:
            if self.state["busy"] and self.stop_file:
                self.stop_file.touch()
                self.state["stop_requested"] = True


def handler(jobs, token, default_engine=None):
    page = Path(__file__).parent / "web/index.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, status=200, content_type="application/json"):
            body = json.dumps(data, ensure_ascii=False).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == host
                    and self.headers.get("Origin", "http://" + host) == "http://" + host
                    and secrets.compare_digest(self.headers.get("X-OpenSV-Token", ""), token))

        def do_GET(self):
            if self.path == "/" and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}":
                return self.reply(page.read_bytes(), content_type="text/html; charset=utf-8")
            if not self.allowed():
                return self.reply({"error": "拒绝访问"}, 403)
            if self.path == "/api/state":
                return self.reply(jobs.snapshot())
            if self.path == "/api/defaults":
                root = Path(__file__).resolve().parents[2]
                name = "opensv.exe" if os.name == "nt" else "opensv"
                engine = default_engine or next((p for p in (root / name, root / "target/release" / name) if p.is_file()), None)
                return self.reply({"engine": str(engine) if engine else shutil.which("opensv") or "", **device_options()})
            if urlsplit(self.path).path == "/api/audio":
                state = jobs.snapshot()
                audio = state.get("audio")
                segment = parse_qs(urlsplit(self.path).query).get("segment", [None])[0]
                if segment is not None:
                    ids = [s["id"] for s in (state.get("preparation") or {}).get("segments", []) if s.get("split")]
                    if segment not in ids:
                        return self.reply({"error": "片段不存在"}, 404)
                    audio = str(Path(state["output"]) / "dataset" / (segment + ".wav"))
                if audio:
                    return self.reply(Path(audio).read_bytes(), content_type="audio/wav")
            return self.reply({"error": "不存在"}, 404)

        def do_POST(self):
            if not self.allowed():
                return self.reply({"error": "拒绝访问"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("请求大小无效")
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict):
                    raise ValueError("请求必须为对象")
                if self.path in ("/api/start", "/api/resume", "/api/prepare"):
                    jobs.start(values, resume=self.path.endswith("resume"), prepare=self.path.endswith("prepare"))
                elif self.path == "/api/stop":
                    jobs.stop()
                elif self.path == "/api/engine":
                    path = shutil.which(values.get("engine", "")) or values.get("engine", "")
                    if not Path(path).is_file():
                        raise ValueError("未找到引擎，请填写编译好的 opensv 程序路径")
                    result = subprocess.run([path, "dataset", "condition", "--help"], capture_output=True,
                                            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if result.returncode != 0 or b"--phrase-seconds" not in result.stdout:
                        raise ValueError("该引擎不支持当前数据准备接口，请更新 OpenSV 引擎")
                    return self.reply({"message": "引擎可用，支持训练数据准备"})
                else:
                    return self.reply({"error": "不存在"}, 404)
                self.reply(jobs.snapshot())
            except (ValueError, OSError, TypeError, subprocess.TimeoutExpired) as error:
                self.reply({"error": str(error)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="OpenSV 本地训练界面")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--engine", type=Path, help="界面默认使用的 OpenSV 引擎")
    args = parser.parse_args()
    if args.engine:
        args.engine = args.engine.expanduser().resolve()
        if not args.engine.is_file():
            parser.error("找不到指定的 OpenSV 引擎")
    token = secrets.token_urlsafe(32)
    jobs = Jobs()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(jobs, token, args.engine))
    url = f"http://127.0.0.1:{server.server_port}/#" + token
    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        jobs.stop()
        print("正在保存训练进度；材料检查或渲染会在当前阶段结束后退出。", flush=True)
        if jobs.worker:
            jobs.worker.join()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

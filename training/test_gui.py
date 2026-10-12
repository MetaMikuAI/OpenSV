import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from opensv_train.gui import Jobs, configuration, device_options, handler, training_target


class GuiTests(unittest.TestCase):
    def write_dataset(self, root):
        root.mkdir(parents=True, exist_ok=True)
        samples = [dict(id=str(i), split=split, features=f"{i}.npz", audio=f"{i}.wav")
                   for i, split in enumerate(("train", "train", "train", "validation"))]
        for name in ["base_model.dnni", "pitch_model.dnni"] + [s[k] for s in samples for k in ("features", "audio")]:
            (root / name).touch()
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps(dict(version=1, supervision="audio", samples=samples)), encoding="utf-8")
        return dataset

    def test_device_options_report_pytorch_capabilities(self):
        for available, version in ((True, "12.8"), (False, "12.8"), (False, None)):
            torch = SimpleNamespace(version=SimpleNamespace(cuda=version),
                                    cuda=SimpleNamespace(is_available=lambda: available),
                                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
            with self.subTest(available=available, version=version), patch.dict("sys.modules", {"torch": torch}):
                result = device_options()
            self.assertEqual(result["cuda_available"], available)
            self.assertEqual(result["cuda_version"], version)
            self.assertEqual(result["devices"], ["cpu", "cuda"] if available else ["cpu"])
            self.assertEqual(bool(result["cuda_reason"]), not available)
        with patch.dict("sys.modules", {"torch": None}):
            self.assertIn("加载失败", device_options()["cuda_reason"])

    def test_unavailable_cuda_is_rejected_before_creating_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"
            jobs = Jobs()
            with patch("opensv_train.gui.device_options", return_value=dict(cuda_available=False, cuda_reason="CPU only")), \
                    patch("opensv_train.gui.threading.Thread") as worker:
                with self.assertRaisesRegex(ValueError, "CUDA 不可用：CPU only"):
                    jobs.start(dict(output=str(output), device="cuda", mode="dataset"))
                worker.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(jobs.snapshot()["busy"])

    def test_mps_device_options(self):
        torch = SimpleNamespace(version=SimpleNamespace(cuda=None),
                                cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict("sys.modules", {"torch": torch}):
            result = device_options()
        self.assertEqual(result["devices"], ["cpu", "mps"])
        self.assertTrue(result["mps_available"])
        self.assertEqual(result["mps_reason"], "")
        self.assertEqual(configuration(dict(device="mps", output="job"))["device"], "mps")

    def test_unavailable_mps_is_rejected_before_creating_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"
            jobs = Jobs()
            with patch("opensv_train.gui.device_options", return_value=dict(mps_available=False, mps_reason="No Metal")), \
                    patch("opensv_train.gui.threading.Thread") as worker:
                with self.assertRaisesRegex(ValueError, "MPS 不可用：No Metal"):
                    jobs.start(dict(output=str(output), device="mps", mode="dataset"))
                worker.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(jobs.snapshot()["busy"])

    def test_metrics_survive_checkpoint_events(self):
        jobs = Jobs()
        jobs.log('{"event":"ready","train_samples":3,"step":0}')
        jobs.log('{"event":"step","step":3,"loss":2.5}')
        jobs.log('{"event":"checkpoint","step":3,"validation_loss":2.7,"best_step":3}')
        state = jobs.snapshot()
        self.assertEqual(state["metrics"]["loss"], 2.5)
        self.assertEqual(state["metrics"]["validation_loss"], 2.7)
        self.assertEqual(state["metrics"]["train_samples"], 3)
        self.assertEqual(state["metrics"]["loss_average"], 2.5)
        self.assertEqual(len(state["history"]), 2)

    def test_raw_loss_survives_history_downsampling(self):
        jobs = Jobs()
        for step in range(1, 2022):
            jobs.log(json.dumps(dict(event="step", step=step, loss=float(step))))
        state = jobs.snapshot()
        self.assertLess(len(state["history"]), 2000)
        self.assertEqual(state["metrics"]["loss"], 2021)
        self.assertEqual(state["metrics"]["loss_average"], 2011.5)
        self.assertEqual(state["history"][-1]["loss"], 2021)

    def test_raw_history_persists_and_replays_checkpoint_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training-history.jsonl"
            jobs = Jobs()
            jobs.history_path = path
            for event in [dict(event="step", step=1, loss=10),
                          dict(event="checkpoint", step=1, validation_loss=5),
                          dict(event="step", step=2, loss=20),
                          dict(event="ready", step=1, train_samples=3),
                          dict(event="step", step=2, loss=7)]:
                jobs.log(json.dumps(event))
            restored = Jobs()
            before = path.read_bytes()
            for line in path.read_text().splitlines():
                restored.log(line, persist=False)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(restored.snapshot()["history"], jobs.snapshot()["history"])
            self.assertEqual([p["loss"] for p in restored.snapshot()["history"] if "loss" in p], [10, 7])
            self.assertEqual(restored.snapshot()["metrics"]["loss"], 7)
            self.assertEqual(restored.snapshot()["metrics"]["validation_loss"], 5)

    def test_history_append_recovers_after_truncated_record(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs()
            jobs.history_path = Path(directory) / "training-history.jsonl"
            jobs.history_path.write_text('{"event":', encoding="utf-8")
            jobs.log('{"event":"step","step":2,"loss":3}')
            restored = Jobs()
            for line in jobs.history_path.read_text().splitlines():
                restored.log(line, persist=False)
            self.assertEqual(restored.snapshot()["history"], [dict(step=2, loss=3, loss_average=3)])

    def test_resume_migrates_same_directory_in_memory_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            dataset = self.write_dataset(root / "dataset")
            engine = root / "engine"
            engine.touch()
            (root / "run").mkdir()
            (root / "run/latest.pt").touch()
            config = configuration(dict(output=str(root), engine=str(engine), voice=str(root),
                                        dataset=str(dataset), epochs=None, steps=6))
            (root / "gui.json").write_text(json.dumps(config), encoding="utf-8")
            jobs = Jobs()
            jobs.update(output=str(root))
            jobs.log('{"event":"step","step":3,"loss":8}')
            with patch("opensv_train.gui.threading.Thread"):
                jobs.start(dict(output=str(root), epochs=None, steps=7), resume=True)
            self.assertEqual(jobs.snapshot()["history"], [dict(step=3, loss=8, loss_average=8)])
            saved = (root / "training-history.jsonl").read_text()
            self.assertEqual(json.loads(saved)["loss"], 8)
            jobs.log('{"event":"ready","step":3,"train_samples":3}')
            jobs.log('{"event":"step","step":4,"loss":2}')
            self.assertEqual([p["loss"] for p in jobs.snapshot()["history"]], [8, 2])

    def test_smoothing_restores_full_window_after_checkpoint_rollback(self):
        jobs = Jobs()
        events = [dict(event="step", step=n, loss=float(n)) for n in range(1, 2101)]
        events += [dict(event="checkpoint", step=2100, validation_loss=1)]
        events += [dict(event="step", step=n, loss=9999.) for n in range(2101, 2151)]
        events += [dict(event="ready", step=2100), dict(event="step", step=2101, loss=2101.)]
        with tempfile.TemporaryDirectory() as directory:
            jobs.history_path = Path(directory) / "training-history.jsonl"
            for event in events:
                jobs.log(json.dumps(event))
            restored = Jobs()
            for line in jobs.history_path.read_text().splitlines():
                restored.log(line, persist=False)
        for state in (jobs.snapshot(), restored.snapshot()):
            self.assertEqual(state["metrics"]["loss"], 2101)
            self.assertEqual(state["metrics"]["loss_average"], 2091.5)
            self.assertLess(len(state["history"]), 2000)
            self.assertEqual(state["history"][-1]["loss_average"], 2091.5)

    def test_smoothing_ignores_checkpoints_and_restarts_after_missing_steps(self):
        jobs = Jobs()
        jobs.log('{"event":"step","step":1,"loss":2}')
        jobs.log('{"event":"checkpoint","step":1,"validation_loss":100}')
        jobs.log('{"event":"step","step":2,"loss":6}')
        self.assertEqual(jobs.snapshot()["metrics"]["loss_average"], 4)
        jobs.log('{"event":"step","step":4,"loss":10}')
        self.assertEqual(jobs.snapshot()["metrics"]["loss_average"], 10)

    def test_epochs_count_only_training_samples(self):
        config = configuration(dict(output="run", epochs=2))
        manifest = dict(supervision="audio", samples=[dict(split=split) for split in
                                                     ("train", "validation", "train", "train")])
        target = training_target(config, manifest)
        self.assertEqual(target["steps"], 6)
        self.assertEqual(target["train_samples"], 3)
        self.assertEqual(target["total_epochs"], 2)
        self.assertEqual(training_target(dict(config, epochs=1.5), manifest)["steps"], 5)
        self.assertEqual(training_target(dict(config, epochs=None, steps=7), manifest)["steps"], 7)

    def test_training_target_rejects_missing_samples_and_excessive_steps(self):
        config = configuration(dict(output="run", epochs=2))
        with self.assertRaisesRegex(ValueError, "没有训练片段"):
            training_target(config, dict(supervision="audio", samples=[dict(split="validation")]))
        for epochs in (10000001, 1e308):
            with self.assertRaisesRegex(ValueError, "不能超过"):
                training_target(dict(config, epochs=epochs),
                                dict(supervision="audio", samples=[dict(split="train")] * 3))

    def test_pair_preparation_records_target_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset.json").write_text(json.dumps(dict(
                supervision="audio", samples=[dict(split="train")] * 3 + [dict(split="validation")]
            )), encoding="utf-8")
            (dataset / "preparation.json").write_text('{"errors":[],"segments":[]}', encoding="utf-8")
            config = configuration(dict(output=directory, epochs=2, mode="pair"))
            jobs = Jobs()
            with patch.object(jobs, "command", return_value=0) as command:
                jobs.run(config, resume=False, prepare=True)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(command.call_args.args[1], "preparing")
            saved = json.loads((root / "gui.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["steps"], 6)
            self.assertEqual(saved["train_samples"], 3)
            self.assertEqual(jobs.snapshot()["phase"], "review")
            self.assertEqual(jobs.snapshot()["config"], saved)

    def test_resume_uses_total_epochs_and_preserves_legacy_steps(self):
        for epochs, steps, previous_device, device in ((3, 9, "cpu", "cuda"), (3, 9, "cpu", "mps"), (None, 7, "mps", "cpu"), (None, 7, "cuda", "cpu"),
                                                       (None, 7, "cuda", None), (None, 7, None, None)):
            with self.subTest(epochs=epochs, device=device, previous_device=previous_device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                engine = root / "engine"
                engine.touch()
                dataset = self.write_dataset(root)
                (root / "run").mkdir()
                (root / "run/latest.pt").touch()
                config = configuration(dict(output=directory, engine=str(engine), voice=directory,
                                            dataset=str(dataset), steps=6, lr=1e-6))
                config.pop("epochs")
                config.pop("preview_language")
                if previous_device:
                    config["device"] = previous_device
                else:
                    config.pop("device")
                (root / "gui.json").write_text(json.dumps(config), encoding="utf-8")
                (root / "training-history.jsonl").write_text('{"event":"step","step":6,"loss":2.5}\n{truncated', encoding="utf-8")
                jobs = Jobs()
                jobs.log('{"event":"step","step":1,"loss":100}')
                values = dict(output=directory, epochs=epochs, steps=7, lr=0.1,
                              engine=str(root / "base_model.dnni"), preview=str(engine), preview_language="japanese",
                              preview_dictionary=str(engine.resolve()))
                if device:
                    values["device"] = device
                with patch("opensv_train.gui.threading.Thread") as worker, \
                        patch("opensv_train.gui.device_options", return_value=dict(cuda_available=True, mps_available=True)):
                    jobs.start(values, resume=True)
                self.assertEqual(jobs.snapshot()["metrics"]["loss"], 2.5)
                self.assertEqual(jobs.snapshot()["history"][0]["step"], 6)
                resumed = worker.call_args.kwargs["args"][0]
                self.assertEqual(resumed["language"], "mandarin")
                self.assertEqual(resumed["preview_language"], "japanese")
                self.assertEqual(resumed["preview"], str(engine.resolve()))
                self.assertEqual(resumed["engine"], str(Path(values["engine"]).resolve()))
                self.assertEqual(resumed["preview_dictionary"], str(engine.resolve()))
                self.assertEqual(resumed["dictionary"], "")
                with patch.object(jobs, "command", return_value=0) as command:
                    jobs.run(resumed, resume=True)
                args = command.call_args_list[0].args[0]
                self.assertIn("--resume", args)
                self.assertEqual(args[args.index("--steps") + 1], str(steps))
                self.assertEqual(args[args.index("--lr") + 1], "1e-06")
                self.assertEqual(args[args.index("--device") + 1], device or previous_device or "cpu")
                saved = json.loads((root / "gui.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["steps"], steps)
                self.assertEqual(saved["train_samples"], 3)
                self.assertEqual(saved["total_epochs"], steps / 3)
                self.assertEqual(jobs.snapshot()["config"], saved)

    def test_preview_language_does_not_change_preparation_language(self):
        for preview_language in ("", "japanese"):
            with self.subTest(preview_language=preview_language), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_dataset(root / "dataset")
                (root / "dataset/preparation.json").write_text('{"errors":[],"segments":[]}', encoding="utf-8")
                config = configuration(dict(output=directory, mode="pair", language="mandarin",
                                            preview="song.svp", preview_language=preview_language, dictionary="dict",
                                            preview_dictionary="preview-dict" if preview_language else ""))
                jobs = Jobs()
                with patch.object(jobs, "command", return_value=0) as command:
                    jobs.run(config, resume=False, prepare=True)
                    args = command.call_args.args[0]
                    self.assertEqual(args[args.index("--language") + 1], "mandarin")
                    self.assertEqual(args[args.index("--dictionary") + 1], "dict")
                with patch.object(jobs, "command", return_value=0) as command:
                    jobs.run(config, resume=False)
                    args, phase = command.call_args.args
                    self.assertEqual(phase, "render")
                    self.assertEqual(args[args.index("--language") + 1], preview_language or "mandarin")
                    self.assertEqual(args[args.index("--dictionary") + 1], "preview-dict" if preview_language else "dict")
                self.assertEqual(config["language"], "mandarin")

    def test_preview_failure_preserves_exported_voice_and_training_status(self):
        for code, phase in ((0, "complete"), (130, "stopped")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                config = configuration(dict(output=directory, preview="song.svp", preview_language="japanese"))
                jobs = Jobs()
                with patch.object(jobs, "command", side_effect=[code, 0, RuntimeError("no pronunciation")]):
                    jobs.run(config, resume=False)
                state = jobs.snapshot()
                self.assertEqual(state["phase"], phase)
                self.assertFalse(state["busy"])
                self.assertNotIn("error", state)
                self.assertNotIn("audio", state)
                self.assertIn("声库已导出，试听渲染失败", state["warning"])
                self.assertIn("no pronunciation", state["warning"])
                self.assertTrue(state["artifacts"][0].endswith(".nofs"))
                self.assertEqual(state["artifacts"][1], str(Path(directory) / "run/trained-base.dnni"))

    def test_invalid_limits(self):
        for values in ({"steps": 0}, {"seconds": -1}, {"lr": "nan"}, {"offset": "nan"}, {"mode": "teacher"},
                       {"epochs": 0}, {"epochs": -1}, {"epochs": "nan"}, {"epochs": "inf"}, {"device": "gpu"},
                       {"preview_language": "unknown"}):
            with self.assertRaises(ValueError):
                configuration({"output": "run", **values})

    def test_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs()
            existing = Path(directory) / "existing"
            existing.write_text("keep", encoding="utf-8")
            with patch("opensv_train.gui.threading.Thread") as worker, patch("opensv_train.gui.subprocess.Popen") as process:
                with self.assertRaisesRegex(ValueError, "工作目录已存在，请选择新的工作目录"):
                    jobs.start(dict(output=directory, engine=str(existing), voice=directory, dataset=str(existing)))
                worker.assert_not_called()
                process.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(Path(directory).iterdir()), [existing])
            self.assertFalse(jobs.snapshot()["busy"])

    def test_missing_preparation_prevents_training(self):
        for missing in ("gui.json", "dataset/preparation.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "dataset").mkdir()
                config = configuration(dict(output=directory, mode="pair"))
                files = {"gui.json": json.dumps(config), "dataset/preparation.json": '{"errors":[]}'}
                for name, content in files.items():
                    if name != missing:
                        (root / name).write_text(content, encoding="utf-8")
                jobs = Jobs()
                with patch("opensv_train.gui.threading.Thread") as worker, patch("opensv_train.gui.subprocess.Popen") as process:
                    with self.assertRaisesRegex(ValueError, "请先点击「检查并切分」，完成后确认对齐，再开始训练"):
                        jobs.start(dict(config, confirm_alignment=True))
                    worker.assert_not_called()
                    process.assert_not_called()
                self.assertFalse((root / missing).exists())
                self.assertFalse((root / "run").exists())
                self.assertFalse((root / "stop.request").exists())
                self.assertFalse(jobs.snapshot()["busy"])
                for name, content in files.items():
                    if name != missing:
                        self.assertEqual((root / name).read_text(encoding="utf-8"), content)

    def test_training_before_splitting_leaves_directory_available(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.touch()
            root = Path(directory) / "job"
            values = dict(output=str(root), engine=str(source), voice=directory,
                          project=str(source), audio=str(source), mode="pair")
            jobs = Jobs()
            with patch("opensv_train.gui.threading.Thread") as worker:
                with self.assertRaisesRegex(ValueError, "请先点击「检查并切分」"):
                    jobs.start(values)
                worker.assert_not_called()
                self.assertFalse(root.exists())
                jobs.start(values, prepare=True)
                self.assertTrue(worker.call_args.kwargs["args"][2])
                worker.return_value.start.assert_called_once()
            self.assertEqual(list(root.iterdir()), [root / "gui.json"])

    def test_incomplete_datasets_do_not_write_job_files(self):
        for mode in ("dataset", "pair", "resume"):
            for problem in ("dataset.json", "base_model.dnni", "pitch_model.dnni", "0.npz", "0.wav",
                            "invalid-json", "invalid-manifest", "no-validation", "excessive-epochs"):
                with self.subTest(mode=mode, problem=problem), tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    source = base / "source"
                    source.touch()
                    root = base / "job"
                    dataset = self.write_dataset(base / "dataset" if mode == "dataset" else root / "dataset")
                    config = configuration(dict(output=str(root), engine=str(source), voice=directory,
                                                project=str(source), audio=str(source), dataset=str(dataset),
                                                mode="dataset" if mode == "dataset" else "pair", epochs=2))
                    if mode != "dataset":
                        (root / "gui.json").write_text(json.dumps(config), encoding="utf-8")
                        (root / "dataset/preparation.json").write_text('{"errors":[]}', encoding="utf-8")
                        (root / "stop.request").touch()
                    if mode == "resume":
                        (root / "run").mkdir()
                        (root / "run/latest.pt").write_bytes(b"checkpoint")
                    if problem == "invalid-json":
                        dataset.write_text("{", encoding="utf-8")
                    elif problem == "invalid-manifest":
                        dataset.write_text("{}", encoding="utf-8")
                    elif problem == "no-validation":
                        manifest = json.loads(dataset.read_text(encoding="utf-8"))
                        manifest["samples"] = manifest["samples"][:3]
                        dataset.write_text(json.dumps(manifest), encoding="utf-8")
                    elif problem == "excessive-epochs":
                        config["epochs"] = 10000000
                    else:
                        (dataset.parent / problem).unlink()
                    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                    jobs = Jobs()
                    with patch("opensv_train.gui.threading.Thread") as worker:
                        with self.assertRaises(ValueError):
                            jobs.start(dict(config, lr=0.1, confirm_alignment=True), resume=mode == "resume")
                        worker.assert_not_called()
                    self.assertFalse(jobs.snapshot()["busy"])
                    self.assertIsNone(jobs.stop_file)
                    self.assertEqual(root.exists(), mode != "dataset")
                    self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_prepared_pair_can_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.touch()
            root = base / "job"
            dataset = self.write_dataset(root / "dataset")
            config = configuration(dict(output=str(root), engine=str(source), voice=directory,
                                        project=str(source), audio=str(source), mode="pair", epochs=2))
            (root / "gui.json").write_text(json.dumps(config), encoding="utf-8")
            (dataset.parent / "preparation.json").write_text('{"errors":[]}', encoding="utf-8")
            jobs = Jobs()
            with patch("opensv_train.gui.threading.Thread") as worker:
                jobs.start(dict(config, confirm_alignment=True, engine=str(dataset.parent / "base_model.dnni"),
                                preview=str(source), preview_language="japanese", preview_dictionary=str(source.resolve())))
                worker.return_value.start.assert_called_once()
            self.assertTrue(jobs.snapshot()["busy"])
            self.assertEqual(jobs.snapshot()["config"]["steps"], 6)
            self.assertEqual(jobs.snapshot()["config"]["language"], "mandarin")
            self.assertEqual(jobs.snapshot()["config"]["preview_language"], "japanese")
            self.assertEqual(jobs.snapshot()["config"]["preview"], str(source.resolve()))
            self.assertEqual(jobs.snapshot()["config"]["preview_dictionary"], str(source.resolve()))
            self.assertEqual(jobs.snapshot()["config"]["dictionary"], "")
            self.assertEqual(jobs.snapshot()["config"]["engine"], str((dataset.parent / "base_model.dnni").resolve()))
            self.assertFalse((root / "run").exists())

    def test_missing_job_prevents_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-job"
            jobs = Jobs()
            with patch("opensv_train.gui.threading.Thread") as worker, patch("opensv_train.gui.subprocess.Popen") as process:
                with self.assertRaisesRegex(ValueError, "没有可续训记录"):
                    jobs.start(dict(output=str(root)), resume=True)
                worker.assert_not_called()
                process.assert_not_called()
            self.assertFalse(root.exists())
            self.assertFalse(jobs.snapshot()["busy"])

    def test_concurrent_job_rejected(self):
        jobs = Jobs()
        jobs.state["busy"] = True
        with self.assertRaisesRegex(ValueError, "已有任务"):
            jobs.start({})

    def test_command_preserves_reported_error(self):
        with patch("opensv_train.gui.subprocess.Popen") as process:
            child = process.return_value.__enter__.return_value
            child.stdout = ['{"event":"error","message":"missing pronunciation dictionary"}', "additional output"]
            child.wait.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "^missing pronunciation dictionary$"):
                Jobs().command(["prepare"], "preparing")

    def test_command_uses_exit_code_without_reported_error(self):
        with patch("opensv_train.gui.subprocess.Popen") as process:
            child = process.return_value.__enter__.return_value
            child.stdout = ["unexpected failure"]
            child.wait.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "preparing 失败，退出码 1"):
                Jobs().command(["prepare"], "preparing")

    def test_command_preserves_rust_error(self):
        with patch("opensv_train.gui.subprocess.Popen") as process:
            child = process.return_value.__enter__.return_value
            child.stdout = ['Error: no pronunciation for "好"\n']
            child.wait.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'no pronunciation for "好"'):
                Jobs().command(["render"], "render")

    def test_alignment_errors_prevent_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            config = configuration(dict(output=directory, mode="pair", voice=directory,
                                        project=directory, audio=directory))
            (root / "gui.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "dataset/preparation.json").write_text('{"errors":["misaligned"]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "检查未通过"):
                Jobs().start(dict(config, confirm_alignment=True))
            self.assertFalse((root / "run").exists())

    def test_api_requires_local_origin_and_token(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Jobs(), "test-token", Path("selected-engine")))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/state"
            for headers in ({}, {"X-OpenSV-Token": "test-token", "Origin": "https://example.com"}):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, headers=headers))
                self.assertEqual(error.exception.code, 403)
            with urlopen(Request(url, headers={"X-OpenSV-Token": "test-token"})) as response:
                self.assertFalse(json.load(response)["busy"])
            with patch("opensv_train.gui.device_options", return_value=dict(devices=["cpu"], cuda_available=False)):
                with urlopen(Request(url.replace("/state", "/defaults"), headers={"X-OpenSV-Token": "test-token"})) as response:
                    self.assertEqual(json.load(response)["devices"], ["cpu"])
                with urlopen(Request(url.replace("/state", "/defaults"), headers={"X-OpenSV-Token": "test-token"})) as response:
                    self.assertEqual(json.load(response)["engine"], "selected-engine")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_page_is_read_on_each_request(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Jobs(), "test-token"))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/"
            with patch("opensv_train.gui.Path.read_bytes", side_effect=[b"first", b"second"]):
                with urlopen(url) as response:
                    self.assertEqual(response.read(), b"first")
                with urlopen(url) as response:
                    self.assertEqual(response.read(), b"second")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

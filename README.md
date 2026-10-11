# OpenSV

歌声模型命令行工具。Rust 负责采集和渲染，Python／PyTorch 负责训练。

## 安装

```powershell
cargo build --release
uv sync --project training --locked
```

以下 `opensv` 指 `target/release/opensv.exe`。uv 在 `training/.venv` 管理 Python 环境，默认使用 Python 3.13 和 CPU 版 PyTorch，依赖版本记录于 `training/uv.lock`。

## 使用

```powershell
uv run --project training --locked opensv dataset capture --audio --voice base-voice --teacher teacher-voice --train-project train.svp --validation-project validation.svp --output dataset --language mandarin --dictionary path/to/clf-data
uv run --project training --locked opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 100
opensv voice-export --voice base-voice --model runs/demo/trained-base.dnni --output trained.nofs
opensv render --voice trained.nofs --project song.svp --output song.wav --language mandarin --dictionary path/to/clf-data --pitch-mode notes
```

声库输入为已解码 NOFS 或模型目录，工程输入为 SVP。支持普通话、日语及显式音素，输出单声道 WAV。默认训练完整声学核心；采集、训练、导出和渲染均在 OpenSV 内完成。

支持用已对齐干声 WAV 监督声学微调，声码器权重固定。当前使用音符音高，尚不支持任意录音的自动对齐、自动 F0、参数曲线、引用组及加密声库。[训练说明](training/README.md)，参数见 `opensv --help`。

## 目录

- `src/`：Rust CLI、模型读取与推理。
- `training/`：Python 训练包与测试。

## 参考项目

[RynthV](https://github.com/cubeww/rynthv)

## 许可证

[GPL-3.0](LICENSE)

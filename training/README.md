# 训练

在仓库根目录运行 `uv sync --project training --locked`。默认 Python 3.13，使用 CPU 版 PyTorch，依赖版本由 `uv.lock` 固定。`uv run` 会让 CLI 使用项目虚拟环境，也可通过 `--python` 显式指定解释器。

## 声学流程

```powershell
uv run --project training --locked metasv dataset capture --voice base-voice --teacher teacher-voice --train-project train.svp --validation-project validation.svp --output dataset --language mandarin --dictionary path/to/clf-data
uv run --project training --locked metasv train --model dataset/base_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 100 --seconds 1800
metasv voice-export --voice base-voice --model runs/demo/trained-base.dnni --output trained.nofs
metasv render --voice trained.nofs --project song.svp --output song.wav --language mandarin --dictionary path/to/clf-data --pitch-mode notes
```

`--voice` 和 `--teacher` 接受已解码 NOFS 或模型目录。目录需要 `base_model.dnni`、`timing_model.dnni`、`pitch_model.dnni`。工程已填写显式音素时可省略词典。

采集使用底模时长、相同音符音高与噪声，将老师输出转换到底模的归一化范围。训练与验证材料须分开；采集会拒绝重复输入，并核对 Rust／PyTorch 输出。尚不支持 SVP／VSQX 配对 WAV 的对齐和特征提取。

| 训练范围 | 参数 | 默认学习率 |
| --- | --- | --- |
| `acoustic`（默认） | 十个声学子网络、音色向量、语言嵌入 | 0.00001 |
| `embeddings` | speaker／style 共 64 个参数 | 0.001 |

声学训练冻结不影响输出的末级残差参数，其余矩阵转为浮点稠密存储，模型文件会变大。导出保留模板的其他模型，不保证兼容官方宿主。

## 数据格式

采集输出 `base_model.dnni`、`dataset.json`、NPZ 与原始特征。清单中的路径相对于清单目录：

```json
{"version":1,"samples":[
  {"id":"train-01","split":"train","features":"train-01.npz"},
  {"id":"validation-01","split":"validation","features":"validation-01.npz"}
]}
```

| NPZ 数组 | 形状 | 内容 |
| --- | --- | --- |
| `phones` | N × 108 | 底模归一化音素输入 |
| `counts` | N | 音素帧数，正整数，总和为 T |
| `languages` | N | 底模语言索引，整数 |
| `pitch` | T × 1 | 归一化音高 |
| `noise` | T × 4 | 生成目标时的噪声 |
| `target` | T × 71 | 底模范围内的归一化声学目标，不是梅尔频谱 |

N 为音素数，T 为帧数。自动采集的清单还包含底模哈希，训练时会校验。

## 续训与输出

```powershell
uv run --project training --locked metasv train --model dataset/base_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 200 --resume
```

`--steps` 是累计步数；`--seconds` 是本次软时间限制，当前步骤、检查和导出可能超时。默认每 25 步保存，正常结束也保存。续训要求底模、数据和训练设置不变。

- `latest.pt`：最新参数、优化器和采样进度。
- `trained-base.dnni`：验证集最优模型，可能早于最新步骤。
- `report.json`：参数数量、训练进度和验证误差。

同一目录只允许一个训练进程。强制结束后若留下 `run.lock`，确认进程已退出再删除。短程测试仅验证流程，不代表音色质量达标。

## 其他模块

`train-components` 可使用已有中间特征训练时长、自动 F0 和声码器；当前采集入口尚不生成这些特征。参数见 `metasv train-components --help`。

其中 `pitch_model.dnni` 是声码器，`f0_model.dnni` 才是自动 F0。各网络单独计算损失，固定滤波器和归一化统计不训练，不进行端到端波形反传；老师与底模权重相同时允许零更新。未指定 `--acoustic-model` 时沿用底模声学模型。

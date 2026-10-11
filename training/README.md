# 训练

在仓库根目录运行 `uv sync --project training --locked`。默认 Python 3.13，使用 CPU 版 PyTorch，依赖版本由 `uv.lock` 固定。`uv run` 会让 CLI 使用项目虚拟环境，也可通过 `--python` 显式指定解释器。

## 声学流程

### 训练界面

先在仓库根目录执行 `cargo build --release`，然后运行：

```powershell
uv run --project training --locked opensv-gui
```

使用 NVIDIA CUDA 时，每次启动都需加以下参数，再在界面选择 CUDA；省略参数会恢复默认 CPU 依赖：

```powershell
uv run --project training --locked --no-group cpu --extra cuda opensv-gui
```

填写底模、干声 WAV、对应的 SVP／VSQX 和新工作目录，点击“检查并切分”。可添加多组材料，每组独立设置时间偏移。程序生成清单、音频片段与训练／验证划分；查看警告并试听后再开始训练。已有清单可直接载入。路径使用绝对路径，在“引擎与发音词典”中填写编译好的 OpenSV 引擎和发音词典，点击“检查引擎”确认接口可用。

材料语言可选普通话或日语，学习率支持滑条与数值输入。默认填写累计轮数，“高级设置”可切换为累计步数；每轮按训练片段数换算，批次大小固定为 1。轮数、步数和曲线每秒刷新；训练误差显示最近 20 步平均值，不足 20 步时取已有记录，原始每步误差保留在日志。固定验证集的误差在检查点更新。

干声 WAV 声学训练支持 CPU 或 NVIDIA CUDA，默认 CPU；数据准备和声库导出仍使用 CPU。

WAV 应是该工程从零点导出的单人干声，保留前导静音，不带伴奏、混响或叠唱。声音晚于工程时填正偏移，早于工程时填负偏移。支持 VSQX 3／4 的速度与预备小节；SVP 引用组须先合并。当前限单人声轨、10 分钟内工程。

音高从 WAV 提取，音素边界由底模估计，不是逐音素强制对齐。时间范围错误、明显错位或片段不足会阻止训练；音高差异、疑似错位和声道差异会提示。检查不能证明歌词、效果器或每个音素都正确。单组按片段约 8:2 划分并跳过边界重叠片段；多组按整组划分，最后约 20% 的材料用于验证。所有材料应属于同一目标声音，重复音频会报错。

“保存并停止”在当前训练步骤结束后保存；材料检查需等当前阶段结束。续训填写原工作目录和更大的累计目标轮数或步数，可切换 CPU／CUDA，材料与其他训练设置沿用 `gui.json`。结果包含检查点、声学模型与 NOFS。关闭网页不停止任务，终端 Ctrl+C 等待保存后退出。

### 命令行

```powershell
uv run --project training --locked opensv-prepare --engine target/release/opensv.exe --voice base-voice --project song.vsqx --audio song.wav --output dataset --dictionary path/to/clf-data
uv run --project training --locked opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 100 --seconds 1800
opensv voice-export --voice base-voice --model runs/demo/trained-base.dnni --output trained.nofs
opensv render --voice trained.nofs --project song.svp --output song.wav --language mandarin --dictionary path/to/clf-data --pitch-mode notes
```

上述 WAV 声学训练使用 CUDA 时，在 `uv run` 参数中添加 `--no-group cpu --extra cuda`，在 `opensv train` 参数中添加 `--device cuda`。CLI 使用累计 `--steps`。

`--voice` 接受已解码 NOFS 或模型目录，目录需要 `base_model.dnni`、`timing_model.dnni`、`pitch_model.dnni`。VSQX 按歌词查底模词典，不复用 VOCALOID 音素编码。SVP 已填写底模支持的显式音素时可省略词典。`--offset` 单位为秒，默认 0。

`--dictionary` 或界面的“发音词典”接受 `clf-data` 目录或自定义 JSON：

```json
{"base":"clf-data","languages":{"japanese":{"人":["h i","t o"]}}}
```

`base` 可省略，只能指向目录，相对路径以 JSON 所在目录为准。用户词条覆盖基础词典；每个数组项是一个音节，音素用空格分隔。汉字读音需自行指定。渲染支持 `+` 分音节、`-`／`ー` 延音和 `br` 呼吸。

界面可单独填写“试听词典”，留空沿用材料词典，不改变已准备的训练材料。

训练将干声的多尺度频谱误差通过固定声码器反传到声学网络，不需要老师模型或中间特征。时长、F0、声码器权重及清浊音硬判定不训练。

导入自动转换双声道和采样率，输出 `dataset.json`、NPZ、单声道 WAV 与 `preparation.json`。CLI 请先查看检查报告；有错误时不生成清单并返回非零退出码。自由演唱录音仍需先对齐到工程时间。

CLI 批量导入以 `--pairs pairs.json` 替代 `--project`／`--audio`。清单格式为 `{"pairs":[{"project":"工程绝对路径","audio":"WAV绝对路径","offset":0}]}`，可添加多组。

| 训练范围 | 参数 | 默认学习率 |
| --- | --- | --- |
| `acoustic`（默认） | 十个声学子网络、音色向量、语言嵌入 | WAV：0.000001；特征：0.00001 |
| `embeddings` | speaker／style 共 64 个参数 | 0.001 |

声学训练冻结不影响输出的末级残差参数，其余矩阵转为浮点稠密存储，模型文件会变大。导出保留模板的其他模型，不保证兼容官方宿主。

## 数据格式

音频采集输出底模、声码器、清单、NPZ 和干声 WAV。清单中的路径相对于清单目录：

```json
{"version":1,"supervision":"audio","samples":[
  {"id":"train-01","split":"train","features":"train-01.npz","audio":"train-01.wav"},
  {"id":"validation-01","split":"validation","features":"validation-01.npz","audio":"validation-01.wav"}
]}
```

| NPZ 数组 | 形状 | 内容 |
| --- | --- | --- |
| `phones` | N × 108 | 底模归一化音素输入 |
| `counts` | N | 音素帧数，正整数，总和为 T |
| `languages` | N | 底模语言索引，整数 |
| `pitch` | T × 1 | 归一化音高 |
| `noise` | T × 4 | 声学推理噪声 |
| `log_f0` | T × 1 | 以 Hz 表示的 F0 的自然对数 |

N 为音素数，T 为帧数，每帧 5 ms。WAV 长度必须等于 T × 声码器步长；支持浮点 32 位、PCM 16／24 位。自动采集还校验底模哈希和 Rust／PyTorch 声学输出。

特征监督：采集省略 `--audio`，训练省略 `--vocoder`，NPZ 使用 `target`（T × 71 的归一化声学目标）替代 `log_f0` 和 WAV。

## 续训与输出

```powershell
uv run --project training --locked opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 200 --resume
```

`--steps` 是累计步数；`--seconds` 是本次软时间限制，当前步骤、检查和导出可能超时。默认每 25 步保存，正常结束也保存。续训要求底模、数据和其他训练设置不变，但可用 `--device cpu` 或 `--device cuda` 切换设备。

- `latest.pt`：最新参数、优化器和采样进度。
- `trained-base.dnni`：验证集最优模型，可能早于最新步骤。
- `report.json`：参数数量、训练进度和验证误差。

同一目录只允许一个训练进程。强制结束后若留下 `run.lock`，确认进程已退出再删除。

## 其他模块

`train-components` 仍使用 CPU，可使用已有中间特征训练时长、自动 F0 和声码器；当前采集入口尚不生成这些特征。参数见 `opensv train-components --help`。

其中 `pitch_model.dnni` 是声码器，`f0_model.dnni` 才是自动 F0。各网络单独计算损失，固定滤波器和归一化统计不训练，不进行端到端波形反传；老师与底模权重相同时允许零更新。未指定 `--acoustic-model` 时沿用底模声学模型。

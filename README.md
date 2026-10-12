# OpenSV

[中文](README.md) | [English](README.en.md)

歌声模型训练与渲染工具，提供命令行和本地训练界面。Rust 负责模型读取与渲染，Python／PyTorch 负责训练。

## 安装

安装 [Rust](https://www.rust-lang.org/tools/install) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在仓库根目录运行：

```powershell
cargo build --release
uv sync --project training --locked
```

uv 在 `training/.venv` 管理 Python 环境，默认 Python 3.13、CPU 版 PyTorch。依赖版本记录于 `training/uv.lock`。已有编译好的引擎时可跳过 Rust 构建，启动后填写引擎路径。

## 训练界面

```powershell
uv run --project training --locked opensv-gui
```

NVIDIA CUDA 用户每次启动使用以下命令，再在界面选择 CPU 或 CUDA。首次运行会下载 CUDA 版 PyTorch；省略这些参数会恢复默认 CPU 依赖。

```powershell
uv run --project training --locked --no-group cpu --extra cuda opensv-gui
```

界面在本机浏览器中运行，可切换中文与英文。界面语言与材料语言分别设置。

1. 填写底模声库和编译好的 `opensv` 引擎，点击“检查引擎”。底模接受 `.svpk` 声库包、`voice.nofs` 或已解码模型目录。支持 `dds` 声学、`gen2a` 时长和 `nhv` 声码器组合；旧版网络和加密 SVDB 不支持。
2. 添加一组或多组干声 WAV 与对应的 SVP／VSQX，填写新工作目录。所有材料应属于同一目标声音，无需手写数据集清单。
3. 选择材料语言（普通话／日语），按需填写底模的发音词典。点击“检查并切分”，查看提示并试听片段。
4. 确认材料对应后设置累计训练轮数并开始训练，也可在高级设置中切换为累计步数。每轮按训练片段数换算为步数，批次大小固定为 1；页面显示轮数及换算后的总步数。

训练误差数值显示当前步原值；图表保留原始曲线，并叠加最近 20 步的平均曲线（不足 20 步时使用已有连续记录）。平滑基于完整逐步记录计算，续训恢复平滑窗口，不受图表抽样影响。逐步误差与检查点验证误差保存至工作目录的 `training-history.jsonl`，续训自动加载历史曲线并继续绘制；长曲线抽样显示，完整原值保留在文件中。验证误差来自固定验证集，在检查点更新，用于观察未参与训练的材料上的表现。

WAV 应从工程零点导出并保留前导静音，不带伴奏、混响或叠唱。每份工程仅接受一条有效歌声轨道，SVP 引用组需先合并；单份工程最长 10 分钟。**音频偏移通常填 0**；若裁掉开头 0.5 秒，填 `-0.5`，若额外添加 0.5 秒静音，填 `0.5`。

干声 WAV 声学训练可选择 CPU、NVIDIA CUDA 或 Apple MPS，默认 CPU；数据准备与声库导出仍使用 CPU。GPU 不一定更快，需按实际数据集测量。时长、自动 F0 和声码器权重保持不变。目标来自干声 WAV，无需老师声库或老师模型的中间特征。音高从 WAV 提取，音素边界由底模估计；它不替代逐音素强制对齐，检查通过也不能保证歌词和每个音素完全对应。

“保存并停止”会在当前步骤结束后保存。续训时填写原工作目录和更大的累计目标轮数或步数（不是本次追加量），再点击“从工作目录续训”；续训可切换 CPU／CUDA／MPS。结果保存在工作目录，包括 `run/latest.pt` 检查点、`run/trained-base.dnni` 声学模型和 `trained-*.nofs` 声库；填写试听工程后还会生成 WAV。关闭网页不会停止任务。

## 命令行

“发音词典”和 `--dictionary` 接受 `clf-data` 目录或自定义 JSON 文件，例如：

```json
{"base":"clf-data","languages":{"japanese":{"人":["h i","t o"]}}}
```

`base` 可省略，但只能指向目录；相对路径以 JSON 所在目录为准。用户词条覆盖基础词典，每个数组项是一个音节，项内音素用空格分隔。汉字读音需自行指定。渲染支持 `+` 分音节、`-`／`ー` 延音和 `br` 呼吸。

界面可单独填写“试听词典”，留空沿用材料词典，不改变已准备的训练材料。

以下 `opensv` 指编译后的引擎，Windows 下为 `target/release/opensv.exe`；可替换为实际程序路径或加入 PATH。

```powershell
uv run --project training --locked opensv-prepare --engine target/release/opensv.exe --voice base.svpk --project song.vsqx --audio song.wav --output dataset --dictionary path/to/clf-data
uv run --project training --locked opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/demo --steps 100
opensv voice-export --voice base.svpk --model runs/demo/trained-base.dnni --output trained.nofs
opensv render --voice trained.nofs --project song.svp --output song.wav --language mandarin --dictionary path/to/clf-data --pitch-mode notes
```

上述 WAV 声学训练使用 CUDA 时，在 `uv run` 参数中添加 `--no-group cpu --extra cuda`，在 `opensv train` 参数中添加 `--device cuda`。

材料导入支持 SVP、VSQX 3／4；直接渲染使用 SVP，支持普通话、日语及显式音素，输出单声道 WAV。详细参数、批量导入及数据格式见[训练说明](training/README.md)和 `opensv --help`。

渲染使用音符音高，尚不支持自动 F0、参数曲线、引用组及加密声库；导出不保证兼容官方宿主。模型、声库、音频和发音词典需自行准备，引擎与分发 ZIP 不提交到 Git。

## Flat 声库装配（实验性）

```powershell
opensv voice-pack --voice trained.nofs --template path/to/voice.104.nofs --models path/to/models --runtime path/to/synthv-flat.dll --name "Example Voice" --vendor "Example Author" --output voice-flat.zip
```

输入声库须包含声学、时长、声码器及自动音高模型；模板及模型目录应来自对应底模的 Flat 安装。命令保留模板模型结构，恢复量化权重并生成默认声音配置，不修改训练文件。`--version` 默认 `201`；可选 `--profile` 和 `--background` 接受 PNG。

ZIP 包含 `databases/` 和 `models/`，解压到 Flat 数据目录后重启宿主。当前只接受已验证的运行库版本，未知版本会报错；不提供官方 SVPK 安装包兼容保证。导出后需在目标宿主试听验证。

## 目录

- `src/`：Rust CLI、模型读取与推理。
- `training/`：Python 训练包与测试。

## 参考项目

- [RynthV](https://github.com/cubeww/rynthv)
- [TrainSV](https://github.com/Quadimension-CL/TrainSV)

## 许可证

[GPL-3.0](LICENSE)

## macOS / Apple MPS

支持 MPS 的 Mac 可使用默认依赖，无需 CUDA extra。训练界面会自动检测并启用“Apple GPU（MPS）”；默认仍选择 CPU。命令行在训练参数中增加 `--device mps`，例如：

```bash
uv run --project training --locked ./target/release/opensv train --model dataset/base_model.dnni --vocoder dataset/pitch_model.dnni --dataset dataset/dataset.json --output runs/mps-demo --steps 100 --device mps
```

MPS 运行声学网络；WAV 监督的固定声码器与频谱损失仍在 CPU 上计算，以保留现有双精度与复数 FFT 算法。跨设备拷贝保留梯度，不会断开训练。特征监督的预测和损失均在 MPS 上计算。数据准备、导出和 `train-components` 仍使用 CPU。CPU／CUDA／MPS 之间可以续训，但不同后端可能存在浮点差异。

此混合路径不保证比 CPU 更快，请用同一数据集比较每步耗时。MPS 不可用或内存不足会明确报错，不会静默切换设备；不需要设置 `PYTORCH_ENABLE_MPS_FALLBACK`。不要关闭 MPS 内存安全限制来规避内存不足。

历史曲线兼容说明：旧版本没有持久化曲线。若同一 GUI 服务仍保留该工作目录的内存历史，续训时会迁移保存；若旧服务已退出且没有 `training-history.jsonl`，无法恢复过去的逐步误差，曲线从后续训练开始记录。

失败任务恢复：同一工作目录可重试。切分失败后修正输入并重新“检查并切分”；训练尚未生成 `run/latest.pt` 时使用“开始训练”；已有检查点时使用“从工作目录续训”。重试前将旧配置、未完成的训练输出及（重新切分时的）旧数据集移动到 `retry-backups/`，不删除材料。普通现有目录、符号链接输出及带 `run.lock` 的目录不会自动重用；遇到锁文件需先确认原进程已经退出。

## 下载包与 CI/CD

Rust CI 检查三个平台的格式、Clippy、测试及构建。下载包包含匹配版本的引擎与 Python 源码：解压后通过 uv 启动，无需编译 Rust 或手动填写引擎路径。Windows 使用 `start.cmd`，macOS ARM64 使用 `start.command`，Linux 使用 `start.sh`。首次启动仍需联网下载 Python 与训练依赖。

维护者可手动运行 **Download bundles** 获取候选包；推送与 Cargo 版本一致的 `v*` 标签会在全部检查和解压启动测试通过后创建 Release 草稿。配置、发布流程与限制见 `packaging/CI.md`。

CD 为每个平台提供 Engine 和 Training 两类下载包，共六个压缩包及对应校验文件。Training 已自带相同版本引擎，无需另下 Engine；仅使用命令行渲染或导出可选择 Engine。

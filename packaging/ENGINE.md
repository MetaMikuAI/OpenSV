# OpenSV Engine

这是独立 Rust 命令行引擎包，包含当前平台的 `opensv`（Windows 为 `opensv.exe`）、许可证、构建信息和对应 Rust 源码。

- 仅需渲染、导出、检查模型：下载 Engine。
- 需要训练或网页界面：下载 Training。Training 已包含相同引擎，无需再下载 Engine。

解压后在终端运行 `./opensv --help`；Windows 使用 `opensv.exe --help`。

示例（自行替换材料路径，Windows 去掉 `./`）：

```bash
./opensv render --voice voice.nofs --project song.svp --output song.wav --language mandarin --dictionary /path/to/clf-data --pitch-mode notes
```

渲染和声库导出不需要 Python；`train`、`train-components` 等委托给 Python 后端的命令不是独立功能，本包不附训练依赖或 GUI。

不含声库、模型、字典、Python 或 uv。请自行准备有权使用的材料。`BUILD.json` 记录版本、平台、源码提交、包类型和引擎哈希；`engine-source/` 为对应源代码。

提供 Windows x64、Linux x64（Ubuntu 22.04 构建基线）和 macOS ARM64 版本。当前未签名／公证，请核对来源和压缩包 SHA-256，不要全局关闭系统安全保护。

## English

Standalone Rust CLI bundle. Rendering, model inspection and voice export do not require Python. Training commands still require the separate Python backend; download the **Training** bundle for the GUI and training workflow. Training already includes the same engine. Models and dictionaries are not included.

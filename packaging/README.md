# OpenSV Training 下载包

这是完整训练包，无需另外下载 Engine 包。仅需命令行渲染或导出时，可选择独立 Engine 包。

包含当前平台的 Rust 引擎、配套 Python 训练源码及锁定依赖清单。无需 Rust 编译器。

1. 安装 uv，并确保终端能够运行 `uv --version`。
2. 完整解压到可写目录，不要直接在压缩包内启动。
3. Windows 双击 `start.cmd`；macOS Apple Silicon 双击 `start.command`；Linux 在终端执行 `./start.sh`。
4. 首次启动由 uv 下载 Python 3.13 和训练依赖；随后打开本机浏览器。引擎路径自动填写为包内的 `opensv` / `opensv.exe`。
5. 填写底模、干声 WAV 和对应 SVP／VSQX，检查切分后开始训练。

## 设备和启动选项

- 默认 CPU；macOS 可在界面选择可用的 MPS。
- Windows NVIDIA CUDA 使用 `start-cuda.cmd`；Linux 使用 `./start.sh --cuda`。每次启动都要使用 CUDA 入口，否则 uv 会同步回默认 CPU 依赖。
- 不打开浏览器：`./start.sh --no-browser`，Windows 为 `start.cmd --no-browser`。
- 自定义端口：`./start.sh --port 8765`。
- 关闭网页不会停止训练；先保存并停止，再关闭终端。

## 范围与限制

支持 Windows x64、Linux x64（构建及测试基线 Ubuntu 22.04）和 macOS ARM64。当前锁文件的 PyTorch 不包含 macOS Intel wheel，不提供 Intel Mac 下载包。

这是联网安装包，不含 Python、uv、PyTorch 或声库。首次运行需要联网和足够磁盘空间。底模支持范围、材料要求和训练限制见 `training/README.md`。模型、音频和 `clf-data` 发音词典需自行准备。

当前包未做 Windows 代码签名或 macOS 公证；操作系统可能提示未知发布者。请确认下载来源及 SHA-256 校验值，不要全局关闭系统安全保护。

每个包包含 `BUILD.json`（版本、源码提交、目标平台、引擎哈希）、`LICENSE` 及 `engine-source/`。升级请解压到新目录，工作数据使用独立目录；不要混用不同版本的引擎和 Python 源码。

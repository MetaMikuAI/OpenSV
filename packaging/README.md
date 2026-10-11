# OpenSV

Windows x64 训练界面，已附 Rust 引擎，无需编译。

1. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，解压完整文件夹。
2. 双击 `start.cmd`。首次启动联网下载 Python 与训练依赖，可能需要几分钟。
3. 填写底模、对应的干声 WAV 和 SVP／VSQX，检查切分结果后开始训练。

可添加多组对应文件，每组只接受单人声轨；多条有效人声轨会拒绝。音素边界由底模估计，请查看警告并试听。

模型、声库、数据集和发音词典需自行准备，本包不包含这些材料。底模需为已解码 NOFS 或模型目录；发音词典在高级设置中填写。详细说明见 [训练说明](training/README.md)。

[GPL-3.0](LICENSE)

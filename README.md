# Microduck RL WebUI

> 基于 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的训练与推理可视化 Web 界面。

---

## 📖 这是什么

**Microduck RL WebUI** 为双足机器人 Microduck 强化学习工作流提供直观的 Web 控制面板：
- **左侧训练监控面板**：配置 PPO 超参数与网络结构（hidden_dims、activation、learning_rate、num_envs、iterations 等），一键拉起训练子进程，实时采集并绘制 Reward / Loss 双轴曲线，支持 ETA 预测与历史日志追溯。
- **右侧推理遥控面板**：加载训练生成的 ONNX 策略模型，利用 MuJoCo 离屏渲染器以 25 FPS 向前端低延迟推流，支持触控摇杆、鼠标及键盘（WASD / QE / 空格）全向速度遥控与姿态重置。

---

## 🔗 与 microduck_rl 的关系

本项目定位为辅助交互层，**不重写或侵入修改** `microduck_rl` 原有物理仿真与 PPO 训练算法，而是对其 CLI (`train` / `scripts/export.py` / `scripts/infer_policy.py`) 进行进程化、服务化封装：
- 训练完全由 `mjlab` (MuJoCo Warp + RSL-RL PPO) 执行。
- 策略推理复用 `onnxruntime` CPU 执行器与 `src/mjlab_microduck/robot/microduck/scene.xml` 真实动力学模型。
- 通过独立解耦的 `PolicyRunner` 剥离对本地 GUI 窗口的依赖，适配云端服务器或远程无显示器工作站环境。

---

## 💻 系统要求

- **操作系统**：Windows 10/11 或 Linux (x86_64 / aarch64)
- **Python 环境**：[uv](https://docs.astral.sh/uv/)（直接复用 `microduck_rl` 已配好的 Python 3.12 虚拟环境）
- **算力设备**：
  - **训练**：推荐配备一块 NVIDIA CUDA GPU（MuJoCo Warp 训练要求）
  - **推理/可视化**：仅需 CPU 即可流畅运行（ONNXRuntime + MuJoCo 离屏渲染）
- **Node.js**：Node.js 18+（开发模式构建前端时使用；仓库已附带打包好的 `frontend/dist`，直接启动即可运行）

---

## 🚀 快速开始

### 1. 准备仓库

确保 `microduck_rl_UI` 与 `microduck_rl` 处于同一父目录下：

```
Project/
├── microduck_rl/         # 原始训练与仿真核心仓库
└── microduck_rl_UI/      # 本 Web 控制面板
```

### 2. 一键启动

在 `microduck_rl_UI` 目录下：

- **Windows 用户**：直接双击运行 `start_webui.bat`
- **命令行用户**：
  ```bash
  # 在 microduck_rl 的 uv 环境中运行
  uv run --directory ../microduck_rl python start_webui.py
  ```

启动成功后，默认在浏览器中自动打开：`http://localhost:8000`

---

## 🕹️ 遥控与按键映射

当右侧"启动仿真推理"开启后，可通过以下方式控制机器人：

| 操作 | 对应按键 | 物理量 / 效果 |
|---|---|---|
| 前进 / 后退 | `W` / `S` 或 摇杆上下 | 线速度 \(v_x\) (\(\pm 0.4\) m/s) |
| 左移 / 右移 | `A` / `D` 或 摇杆左右 | 横向线速度 \(v_y\) (\(\pm 0.3\) m/s) |
| 原地转向 | `Q` / `E` 或 转向按钮 | 角速度 \(w_z\) (\(\pm 0.8\) rad/s) |
| 刹停 / 回中 | `空格键` 或 "回中"按钮 | 所有速度指令置零 (Coast) |
| 仿真重置 | 点击"重置姿态"按钮 | 重置 MuJoCo 物理状态至初始站立位 |
| 暂停 / 继续 | 点击"暂停"按钮 | 保持最后电机力矩，暂停仿真步进 |

---

## 🏛️ 项目结构

```
microduck_rl_UI/
├── backend/
│   ├── app.py               # FastAPI 服务入口 (REST API + WebSockets)
│   ├── config.py            # 路径发现与全局配置
│   ├── db.py                # SQLite 任务与模型持久化
│   ├── schemas.py           # Pydantic 请求模型与校验
│   ├── training_manager.py  # 训练子进程管理与白名单防御
│   ├── metrics_tailer.py    # TensorBoard EventAccumulator 实时解析器
│   ├── policy_runner.py     # 独立 MuJoCo 离屏渲染与 ONNX 推理核心
│   └── inference_manager.py # 视频推流调度与命令队列
├── frontend/                # React + Vite + TypeScript + Tailwind CSS
│   ├── src/                 # 前端组件、面板、图表与 WebSocket Hooks
│   └── dist/                # 编译打包好的生产环境前端静态资源
├── docs/
│   └── architecture.md      # 完整架构设计文档
├── exports/                 # 导出 ONNX 模型存放目录
├── runs/                    # 训练过程日志与元数据
├── webui.db                 # 本地 SQLite 数据库
├── start_webui.py           # Python 启动脚本
├── start_webui.bat          # Windows 快捷启动批处理
├── LICENSE                  # Apache-2.0 全文
└── README.md                # 本使用说明
```

详细架构设计、通信协议与 UI 规范请参阅 [docs/architecture.md](docs/architecture.md)。

---

## ⚠️ 已知限制

1. **GPU 显存独占**：由于 MuJoCo Warp 训练会占用较大显存，后端内置全局任务锁，同一时刻仅允许一个训练子进程运行。
2. **网络结构定义**：当前为参数化网络配置（隐藏层维度、激活函数等），如需实现计算图层级的自由拼接，请参考架构设计文档的 Phase 5 规划。

---

## 📄 许可证与致谢

本项目采用 [Apache-2.0](LICENSE) 许可证。

**致谢上游项目**：
- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache 2.0)
- [mujocolab/mjlab](https://github.com/mujocolab/mjlab)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

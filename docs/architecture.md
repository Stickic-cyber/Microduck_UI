# Microduck RL 训练/推理可视化 WebUI —— 架构设计文档

> 目标仓库：`pollen-robotics/microduck_rl`（mjlab + MuJoCo Warp + PPO，训练 → 导出 ONNX → CPU MuJoCo 推理）
> 需求：一个网页，左侧配置/训练强化学习模型并实时看 loss/reward 曲线，右侧加载训练出的 `.onnx` 做可视化遥控（teleop）展示。

---

## 0. 先对齐一个关键预期

`microduck_rl` 的策略网络是通过 **rsl_rl 的 PPO ActorCritic 配置**（MLP 隐藏层维度、激活函数等超参数）来定义的，不是一个可以任意拖拽连线的计算图。所以"左侧自己搭建神经网络"在工程上分两档，建议分阶段做：

- **档位 A（MVP，强烈建议先做这个）**：网页上提供表单，可配置 `hidden_dims`（如 `[512,256,128]`）、`activation`、`learning_rate`、`num_envs`、`max_iterations` 等 rsl_rl/PPO 超参数，本质是"配置网络"而不是"画网络"。
- **档位 B（进阶，可选，先不做，预留接口）**：提供一个受限的可视化模块拼装器（比如几种预设 block：MLP block、LSTM block、LayerNorm），前端生成一段 Python 网络定义代码片段，后端校验后动态写入训练配置。

---

## 1. 整体架构

```
┌─────────────────────────────┐        WebSocket/REST        ┌───────────────────────────────────────┐
│         前端 (Browser)        │ <───────────────────────────>│         后端 (FastAPI, Python)          │
│  React + Vite + Tailwind     │                               │  与 microduck_rl 同一 Python/uv 环境    │
│                               │                               │                                         │
│  ┌───────────┬─────────────┐ │                               │  ┌───────────────┐  ┌────────────────┐ │
│  │  左：训练面板 │ 右：推理可视化 │ │                               │  │ TrainingManager│  │ InferenceManager│ │
│  │  - 超参表单   │ - 视频流画面   │ │                               │  │ (子进程/多进程)  │  │  (子进程)        │ │
│  │  - 开始/停止  │ - onnx 选择器  │ │                               │  └──────┬────────┘  └────────┬───────┘ │
│  │  - loss曲线  │ - 遥控按键/摇杆 │ │                               │         │                    │          │
│  │  - 迭代计数器 │               │ │                               │  ┌──────▼────────┐  ┌────────▼───────┐ │
│  └───────────┴─────────────┘ │                               │  │ uv run train   │  │ scripts/infer_  │ │
└─────────────────────────────┘                               │  │ (mjlab+PPO)    │  │ policy.py 逻辑  │ │
                                                                 │  └────────────────┘  │ (改为headless   │ │
                                                                 │                       │  render+ws输入) │ │
                                                                 │                       └────────────────┘ │
                                                                 │  metrics tailer (tensorboard event)      │
                                                                 │  job registry / GPU 互斥锁 / SQLite 持久化 │
                                                                 └───────────────────────────────────────┘
```

核心思路：**后端不重写训练/推理逻辑，而是把 `microduck_rl` 现成的 `train` / `export` / `infer_policy.py` 包装成可被 Web 控制、可流式输出状态的服务**。

---

## 2. 技术栈建议

| 层 | 选型 | 理由 |
|---|---|---|
| 前端框架 | React + Vite + TypeScript | 组件化好维护，图表/websocket生态成熟 |
| 前端图表 | recharts 或 lightweight-charts | 实时曲线，支持双 y 轴（reward/loss） |
| 前端样式 | Tailwind CSS | 快速搭左右分栏布局 |
| 后端框架 | FastAPI + Uvicorn | 原生支持 WebSocket，和 Python/uv 训练脚本同栈 |
| 进程管理 | Python `subprocess` / `asyncio` | 训练是 GPU 长任务，必须独立进程，不能阻塞事件循环 |
| 指标采集 | tensorboard event 文件 tail (`EventAccumulator`) | rsl_rl 默认写 tensorboard log，无需侵入训练代码即可读 |
| 状态持久化 | SQLite | 记录历史训练任务、checkpoint、导出的 onnx 列表，前端刷新/断线重连后还能看到 |
| 渲染流 | `mujoco.Renderer` 离屏渲染 → JPEG → base64 over WebSocket | 类似 MJPEG 推流，不需要 WebRTC 那么复杂，够用 |
| 推理 | onnxruntime（CPU） | 复用 `infer_policy.py` 里已有的加载/推理逻辑 |

---

## 3. UI 风格规范

核心原则：**克制、准确、不拟人化**。
- 中性底色（白或浅灰 `#F7F7F8`）+ 1 个主色（用于"训练中/正常"状态，如蓝或绿）+ 1 个警示色（红/橙，仅用于报错/停止）。
- 系统默认无衬线字体；数字类信息（迭代数、reward 值）用等宽字体。
- 状态类信息必须"颜色+文字"同时呈现。
- 术语解释用一句话中性定义，不展开类比故事。

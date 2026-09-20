import { Checkpoint, OnnxModel, TrainConfigForm, TrainStatus } from "./types";

const API_BASE = "/api";

export async function fetchEnvs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/envs`);
  if (!res.ok) throw new Error("获取任务列表失败");
  return res.json();
}

export async function fetchTrainStatus(): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/train/status`);
  if (!res.ok) throw new Error("获取训练状态失败");
  return res.json();
}

export async function startTrain(form: TrainConfigForm): Promise<any> {
  const hidden_dims = form.hidden_dims
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n));

  const payload = {
    env_id: form.env_id,
    num_envs: form.num_envs,
    max_iterations: form.max_iterations,
    run_name: form.run_name || undefined,
    network: {
      hidden_dims: hidden_dims.length > 0 ? hidden_dims : [512, 256, 128],
      activation: form.activation,
    },
    ppo: {
      learning_rate: form.learning_rate,
      gamma: form.gamma,
      entropy_coef: form.entropy_coef,
      clip_param: form.clip_param,
      value_loss_coef: form.value_loss_coef,
    },
  };

  const res = await fetch(`${API_BASE}/train/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "启动训练失败");
  }
  return res.json();
}

export async function stopTrain(): Promise<any> {
  const res = await fetch(`${API_BASE}/train/stop`, { method: "POST" });
  if (!res.ok) throw new Error("停止训练失败");
  return res.json();
}

export async function fetchTrainHistory(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/train/history`);
  if (!res.ok) throw new Error("获取历史记录失败");
  return res.json();
}

export async function fetchTrainLogs(): Promise<{ run_name: string | null; logs: string[] }> {
  const res = await fetch(`${API_BASE}/train/logs`);
  if (!res.ok) throw new Error("获取日志失败");
  return res.json();
}

export async function fetchCheckpoints(runName?: string): Promise<Checkpoint[]> {
  const url = runName ? `${API_BASE}/checkpoints?run_name=${encodeURIComponent(runName)}` : `${API_BASE}/checkpoints`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("获取检查点列表失败");
  return res.json();
}

export async function exportOnnx(taskId: string, checkpointPath: string, onnxFilename?: string, runName?: string): Promise<any> {
  const res = await fetch(`${API_BASE}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task_id: taskId,
      checkpoint_path: checkpointPath,
      onnx_filename: onnxFilename,
      run_name: runName,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "导出 ONNX 失败");
  }
  return res.json();
}

export async function fetchOnnxModels(): Promise<OnnxModel[]> {
  const res = await fetch(`${API_BASE}/onnx/list`);
  if (!res.ok) throw new Error("获取 ONNX 模型列表失败");
  return res.json();
}

export async function fetchInferStatus(): Promise<any> {
  const res = await fetch(`${API_BASE}/infer/status`);
  if (!res.ok) throw new Error("获取推理状态失败");
  return res.json();
}

export async function startInfer(onnxPath: string, fps = 25): Promise<any> {
  const res = await fetch(`${API_BASE}/infer/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ onnx_path: onnxPath, fps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "启动推理失败");
  }
  return res.json();
}

export async function stopInfer(): Promise<any> {
  const res = await fetch(`${API_BASE}/infer/stop`, { method: "POST" });
  if (!res.ok) throw new Error("停止推理失败");
  return res.json();
}

export async function sendInferCommand(vx: number, vy: number, wz: number, action?: string): Promise<any> {
  const res = await fetch(`${API_BASE}/infer/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vx, vy, wz, action }),
  });
  if (!res.ok) throw new Error("发送遥控指令失败");
  return res.json();
}

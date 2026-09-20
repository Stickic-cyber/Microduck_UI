export interface MetricPoint {
  iteration: number;
  mean_reward: number;
  loss: Record<string, number>;
  fps?: number;
  elapsed_s?: number;
  eta_s?: number;
}

export interface TrainStatus {
  state: "idle" | "running" | "finished" | "stopped" | "error";
  run_name?: string | null;
  task_id?: string | null;
  pid?: number | null;
  log_dir?: string | null;
  error_message?: string | null;
  current_iteration?: number;
  max_iterations?: number;
  mean_reward?: number | null;
  eta_str?: string | null;
  elapsed_str?: string | null;
  elapsed_sec?: number | null;
  latest_log?: string | null;
}

export interface TrainConfigForm {
  env_id: string;
  num_envs: number;
  max_iterations: number;
  run_name: string;
  hidden_dims: string;
  activation: string;
  learning_rate: number;
  gamma: number;
  entropy_coef: number;
  clip_param: number;
  value_loss_coef: number;
}

export interface Checkpoint {
  id?: number;
  run_name: string;
  checkpoint_name: string;
  checkpoint_path: string;
  iteration?: number;
  created_at: string;
}

export interface OnnxModel {
  name: string;
  path: string;
  source: string;
}

export interface FrameMessage {
  type: "frame";
  seq: number;
  jpeg_b64: string;
  sim_time: number;
  step: number;
  cmd: {
    vx: number;
    vy: number;
    wz: number;
  };
}

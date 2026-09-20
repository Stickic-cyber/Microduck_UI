"""Pydantic data schemas for requests and responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class NetworkConfig(BaseModel):
    hidden_dims: List[int] = Field(default=[512, 256, 128], description="Actor/critic hidden layer dimensions")
    activation: str = Field(default="elu", description="Activation function (elu, relu, tanh)")

class PPOConfig(BaseModel):
    learning_rate: float = Field(default=0.001, ge=1e-6, le=0.1, description="Learning rate")
    gamma: float = Field(default=0.99, ge=0.5, le=0.9999, description="Discount factor")
    entropy_coef: float = Field(default=0.01, ge=0.0, le=0.5, description="Entropy coefficient")
    clip_param: float = Field(default=0.2, ge=0.01, le=0.5, description="PPO clipping parameter")
    value_loss_coef: float = Field(default=1.0, ge=0.01, le=10.0, description="Value loss coefficient")

class TrainStartRequest(BaseModel):
    env_id: str = Field(default="Mjlab-Velocity-Flat-MicroDuck", description="Target RL task ID")
    num_envs: int = Field(default=64, ge=1, le=8192, description="Parallel environments count")
    max_iterations: int = Field(default=1000, ge=1, le=100000, description="Maximum training iterations")
    run_name: Optional[str] = Field(default=None, description="Custom name for this training run")
    network: NetworkConfig = Field(default_factory=NetworkConfig, description="Policy network architecture")
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO hyperparameters")
    device: Optional[str] = Field(default=None, description="Compute device (cuda:0, cpu)")

class ExportRequest(BaseModel):
    task_id: str = Field(default="Mjlab-Velocity-Flat-MicroDuck", description="Task ID for the checkpoint")
    checkpoint_path: str = Field(..., description="Absolute path or filename of .pt checkpoint")
    onnx_filename: Optional[str] = Field(default=None, description="Target ONNX filename, e.g. policy.onnx")
    run_name: Optional[str] = Field(default=None, description="Associated run name")

class InferStartRequest(BaseModel):
    onnx_path: str = Field(..., description="Absolute path of the ONNX policy to run")
    fps: int = Field(default=25, ge=5, le=60, description="Render frame rate")
    jpeg_quality: int = Field(default=65, ge=20, le=95, description="JPEG compression quality")

class InferCommand(BaseModel):
    vx: float = Field(default=0.0, description="Forward/backward linear velocity (m/s)")
    vy: float = Field(default=0.0, description="Lateral strafe velocity (m/s)")
    wz: float = Field(default=0.0, description="Yaw angular velocity (rad/s)")
    action: Optional[str] = Field(default=None, description="Special action (reset, pause, resume)")

class TrainMetricMessage(BaseModel):
    type: str = "metric"
    iteration: int
    mean_reward: float
    loss: Dict[str, float]
    fps: Optional[float] = None
    elapsed_s: Optional[float] = None
    eta_s: Optional[float] = None

class TrainStatusMessage(BaseModel):
    type: str = "status"
    state: str  # idle, running, finished, stopped, error
    run_name: Optional[str] = None
    message: Optional[str] = None

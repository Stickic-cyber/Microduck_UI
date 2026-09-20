"""PolicyRunner: Headless MuJoCo policy inference and offscreen renderer."""

import base64
import io
import math
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image

from backend.config import SCENE_XML_PATH

# Default reference pose used by the policy (matching microduck_rl standards)
DEFAULT_POSE = np.array([
    0.0,      # left_hip_yaw
    -0.0873,  # left_hip_roll
    -0.4579,  # left_hip_pitch
    -0.0049,  # left_knee
    0.4530,   # left_ankle
    0.3491,   # neck_pitch
    0.3491,   # head_pitch
    0.0,      # head_yaw
    0.0,      # head_roll
    0.0,      # right_hip_yaw
    0.0873,   # right_hip_roll
    0.4579,   # right_hip_pitch
    0.0049,   # right_knee
    -0.4530,  # right_ankle
], dtype=np.float32)

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

class PolicyRunner:
    """Manages MuJoCo simulation, ONNX policy inference, and offscreen rendering.

    Note on Observation Dimensions (Upstream microduck_rl contract):
    - 51D (Legacy):
        ang_vel(3) + gravity(3) + joint_pos(14) + joint_vel(14) + last_action(14) + twist_cmd(3) = 51
    - 61D (Unified Multi-task contract, verified on output_5000.onnx & output_16750.onnx):
        ang_vel(3) + gravity(3) + joint_pos(14) + joint_vel(14) + last_action(14) + command(13) = 61
        where command(13) = twist(3) + head_pose(4) + body_pose(6).
    """

    def __init__(
        self,
        onnx_path: str,
        xml_path: Optional[str] = None,
        render_width: int = 480,
        render_height: int = 360,
        action_scale: float = 1.0,
        decimation: int = 10,
    ):
        self.onnx_path = str(Path(onnx_path).resolve())
        self.xml_path = str(Path(xml_path or SCENE_XML_PATH).resolve())
        self.render_width = render_width
        self.render_height = render_height
        self.action_scale = action_scale
        self.decimation = decimation

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MuJoCo XML scene not found: {self.xml_path}")
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX policy model not found: {self.onnx_path}")

        # Initialize MuJoCo model & data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Offscreen renderer (initialized lazily on the rendering worker thread for GLFW WGL safety)
        self.renderer: Optional[mujoco.Renderer] = None

        # Camera setup (tracking trunk_base)
        self.camera = mujoco.MjvCamera()
        self.trunk_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if self.trunk_base_id >= 0:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.trunk_base_id
            self.camera.distance = 1.05
            self.camera.elevation = -18.0
            self.camera.azimuth = 140.0
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.distance = 1.2

        # Cache joint indices
        self.n_joints = len(JOINT_NAMES)
        self.joint_qpos_indices = []
        self.joint_qvel_indices = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            self.joint_qpos_indices.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_indices.append(self.model.jnt_dofadr[jid])

        self.joint_qpos_indices = np.array(self.joint_qpos_indices, dtype=np.int32)
        self.joint_qvel_indices = np.array(self.joint_qvel_indices, dtype=np.int32)

        # Sensors
        self.imu_ang_vel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.imu_accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")

        # Load ONNX session
        self.ort_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_name = self.ort_session.get_outputs()[0].name
        self.output_shape = self.ort_session.get_outputs()[0].shape
        self.obs_dim = self.input_shape[1] if len(self.input_shape) > 1 else 61

        if self.obs_dim not in (51, 61):
            warnings.warn(
                f"[PolicyRunner] Loaded ONNX has observation dim {self.obs_dim}, which differs from "
                "standard microduck_rl shapes (51 legacy or 61 unified). Command padding will adapt."
            )

        self.is_unified_obs = (self.obs_dim >= 61)

        # State and commands
        self._cmd_lock = threading.Lock()
        self.vel_cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.body_cmd = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.default_pose = DEFAULT_POSE.copy()

        self.step_count = 0
        self.sim_time = 0.0
        self.is_paused = False

        self.reset()

    def init_renderer(self) -> mujoco.Renderer:
        """Initialize MuJoCo offscreen renderer on the current thread context."""
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        return self.renderer

    def close(self) -> None:
        """Release GLFW / OpenGL rendering resources safely."""
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None

    def __del__(self) -> None:
        self.close()

    def reset(self) -> None:
        """Reset physics simulation and policy state."""
        with self._cmd_lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.joint_qpos_indices] = self.default_pose
            mujoco.mj_forward(self.model, self.data)

            self.last_action.fill(0.0)
            self.vel_cmd.fill(0.0)
            self.head_offset.fill(0.0)
            self.body_cmd.fill(0.0)
            self.step_count = 0
            self.sim_time = 0.0

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Update commanded velocities thread-safely."""
        with self._cmd_lock:
            self.vel_cmd[0] = float(np.clip(vx, -1.0, 1.0))
            self.vel_cmd[1] = float(np.clip(vy, -0.6, 0.6))
            self.vel_cmd[2] = float(np.clip(wz, -2.0, 2.0))

    def set_paused(self, paused: bool) -> None:
        self.is_paused = paused

    def quat_rotate_inverse(self, quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def get_base_ang_vel(self) -> np.ndarray:
        if self.imu_ang_vel_id >= 0:
            sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
            return self.data.sensordata[sensor_adr:sensor_adr + 3].copy().astype(np.float32)
        trunk_qvel_adr = self.model.jnt_dofadr[0]
        return self.data.qvel[trunk_qvel_adr + 3:trunk_qvel_adr + 6].copy().astype(np.float32)

    def get_gravity_vector(self) -> np.ndarray:
        if self.trunk_base_id >= 0:
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """Build observation vector matching the policy's input dimensions."""
        base_ang_vel = self.get_base_ang_vel()
        grav_vec = self.get_gravity_vector()
        joint_pos = (self.data.qpos[self.joint_qpos_indices] - self.default_pose).astype(np.float32)
        joint_vel = self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

        with self._cmd_lock:
            if self.is_unified_obs:
                cmd = np.zeros(13, dtype=np.float32)
                cmd[0:3] = self.vel_cmd
                cmd[3:7] = self.head_offset
                cmd[7:13] = self.body_cmd
            else:
                cmd = self.vel_cmd.copy()

        obs_core = np.concatenate([
            base_ang_vel,
            grav_vec,
            joint_pos,
            joint_vel,
            self.last_action,
            cmd,
        ]).astype(np.float32)

        # Pad or trim if obs_dim has unexpected variation
        if len(obs_core) < self.obs_dim:
            padded = np.zeros(self.obs_dim, dtype=np.float32)
            padded[:len(obs_core)] = obs_core
            return padded
        elif len(obs_core) > self.obs_dim:
            return obs_core[:self.obs_dim]

        return obs_core

    def step(self, quality: int = 65) -> Tuple[str, Dict[str, Any]]:
        """Run 1 control step (policy inference + decimation physics) and offscreen render."""
        if not self.is_paused:
            obs = self.get_observation()
            obs_input = obs.reshape(1, -1)

            outputs = self.ort_session.run(None, {self.input_name: obs_input})
            action = np.squeeze(outputs[0]).astype(np.float32)
            self.last_action = action.copy()

            targets = self.default_pose + action * self.action_scale
            self.data.ctrl[:self.n_joints] = targets

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.sim_time = self.data.time

        if self.renderer is None:
            self.init_renderer()

        self.renderer.update_scene(self.data, camera=self.camera)
        rgb_array = self.renderer.render()

        img = Image.fromarray(rgb_array)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        jpeg_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        state_info = {
            "sim_time": round(float(self.sim_time), 2),
            "step": self.step_count,
            "cmd": {
                "vx": round(float(self.vel_cmd[0]), 2),
                "vy": round(float(self.vel_cmd[1]), 2),
                "wz": round(float(self.vel_cmd[2]), 2),
            },
        }

        return jpeg_b64, state_info

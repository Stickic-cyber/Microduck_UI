"""Training Manager: Manages PPO training subprocesses, log extraction, and process exclusivity."""

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import psutil
import torch

from backend import db
from backend.config import MICRODUCK_RL_DIR, RUNS_DIR, UV_BIN
from backend.metrics_tailer import MetricsTailer
from backend.schemas import TrainStartRequest

class TrainingManager:
    """Singleton-style manager for handling training runs."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.current_process: Optional[subprocess.Popen] = None
        self.current_pid: Optional[int] = None
        self.current_run_name: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.current_log_dir: Optional[Path] = None
        self.current_status: str = "idle"  # idle, running, finished, stopped, error
        self.current_error: Optional[str] = None
        self.metrics_tailer: Optional[MetricsTailer] = None
        self._tailer_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self.log_lines: Deque[str] = deque(maxlen=1000)
        self.log_file_path: Optional[Path] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Realtime progress tracking from stdout & tailer
        self.current_iter: int = 0
        self.max_iter: int = 100
        self.mean_reward: Optional[float] = None
        self.eta_str: Optional[str] = None
        self.elapsed_str: Optional[str] = None
        self.latest_log_summary: str = "空闲待命"
        self.start_timestamp: Optional[float] = None

        # Check for un-finished jobs on startup (process recovery)
        self.adopt_existing_job()

    def _start_metrics_tailer(self, log_dir: Path, max_iterations: int) -> None:
        """Start MetricsTailer in event loop context, replacing any prior instance."""
        if self.metrics_tailer:
            try:
                self.metrics_tailer.stop()
            except Exception:
                pass
            self.metrics_tailer = None

        self.metrics_tailer = MetricsTailer(
            log_dir=log_dir,
            max_iterations=max_iterations,
        )
        self._tailer_task = asyncio.create_task(self.metrics_tailer.run_loop())
        print(f"[TrainingManager] MetricsTailer scheduled on event loop for log_dir: {log_dir}", flush=True)

    def adopt_existing_job(self) -> None:
        """Adopts an ongoing training job if FastAPI restarted."""
        latest = db.get_latest_train_run()
        if latest and latest["status"] == "running":
            pid = latest.get("pid")
            if pid and psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    if "python" in proc.name().lower() or "train" in proc.name().lower():
                        print(f"[TrainingManager] Adopting active training run '{latest['run_name']}' (PID: {pid})")
                        self.current_pid = pid
                        self.current_run_name = latest["run_name"]
                        self.current_task_id = latest["task_id"]
                        self.current_status = "running"
                        if latest.get("log_dir"):
                            self.current_log_dir = Path(latest["log_dir"])
                            max_it = latest.get("config", {}).get("max_iterations", 1000)
                            self.metrics_tailer = MetricsTailer(log_dir=self.current_log_dir, max_iterations=max_it)
                        return
                except Exception:
                    pass
            # Process died while backend was off
            db.update_train_run_status(latest["run_name"], "stopped", error_message="FastAPI 服务重启时训练进程已终止")

    def is_running(self) -> bool:
        if self.current_process is not None:
            return self.current_process.poll() is None
        if self.current_pid is not None:
            return psutil.pid_exists(self.current_pid)
        return False

    def get_status(self) -> Dict[str, Any]:
        running = self.is_running()
        state = "running" if running else self.current_status
        pid = self.current_process.pid if self.current_process else self.current_pid
        elapsed_sec = round(time.time() - self.start_timestamp, 1) if self.start_timestamp and running else None
        return {
            "state": state,
            "run_name": self.current_run_name,
            "task_id": self.current_task_id,
            "log_dir": str(self.current_log_dir) if self.current_log_dir else None,
            "pid": pid,
            "error_message": self.current_error,
            "current_iteration": self.current_iter,
            "max_iterations": self.max_iter,
            "mean_reward": self.mean_reward,
            "eta_str": self.eta_str,
            "elapsed_str": self.elapsed_str,
            "elapsed_sec": elapsed_sec,
            "latest_log": self.latest_log_summary,
        }

    def get_logs(self, limit: int = 100) -> List[str]:
        # If in-memory buffer has lines, return them
        if self.log_lines:
            return list(self.log_lines)[-limit:]
        # Otherwise tail the log file from disk
        if self.log_file_path and self.log_file_path.exists():
            try:
                with open(self.log_file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = [line.rstrip() for line in f.readlines()]
                    return lines[-limit:]
            except Exception:
                pass
        return []

    async def start_train(self, req: TrainStartRequest) -> Dict[str, Any]:
        async with self._lock:
            if self.is_running():
                raise RuntimeError("当前已有训练任务正在运行中，请勿重复启动")

            # Check database for active runs across any workers
            active_runs = [r for r in db.get_train_runs(limit=5) if r["status"] == "running"]
            for ar in active_runs:
                if ar["pid"] and psutil.pid_exists(ar["pid"]):
                    raise RuntimeError(f"任务 '{ar['run_name']}' (PID: {ar['pid']}) 仍在运行中")
                else:
                    db.update_train_run_status(ar["run_name"], "stopped")

            # Validate whitelist
            env_id = re.sub(r"[^\w\-]", "", req.env_id)
            if not env_id:
                raise ValueError("无效的环境标识符 (env_id)")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = req.run_name.strip() if req.run_name else f"run_{timestamp}"
            run_name = re.sub(r"[^\w\-]", "_", run_name)

            # Reset prior tailer and tasks
            if self.metrics_tailer:
                try:
                    self.metrics_tailer.stop()
                except Exception:
                    pass
                self.metrics_tailer = None
            if self._tailer_task and not self._tailer_task.done():
                self._tailer_task.cancel()
                self._tailer_task = None

            self.current_run_name = run_name
            self.current_task_id = env_id
            self.current_status = "running"
            self.current_error = None
            self.current_log_dir = None
            self.current_iter = 0
            self.max_iter = req.max_iterations
            self.mean_reward = None
            self.eta_str = None
            self.elapsed_str = None
            self.latest_log_summary = "正在启动 MuJoCo Warp 并行环境..."
            self.start_timestamp = time.time()
            self.log_lines.clear()
            self._loop = asyncio.get_running_loop()

            self.log_file_path = RUNS_DIR / f"{run_name}.log"

            # Determine GPU IDs based on hardware detection
            cuda_available = torch.cuda.is_available()
            if req.device == "cpu" or not cuda_available:
                gpu_id_arg = "None"
            elif req.device and req.device.isdigit():
                gpu_id_arg = str(req.device)
            else:
                gpu_id_arg = "0"

            # Format hidden-dims as tuple string literal like '(512, 256, 128)' for tyro CLI parser
            hidden_dims_tuple_str = str(tuple(req.network.hidden_dims))

            # Construct CLI arguments
            cmd = [
                UV_BIN,
                "run",
                "train",
                env_id,
                "--gpu-ids",
                gpu_id_arg,
                "--env.scene.num-envs",
                str(req.num_envs),
                "--agent.actor.hidden-dims",
                hidden_dims_tuple_str,
                "--agent.actor.activation",
                req.network.activation,
                "--agent.critic.hidden-dims",
                hidden_dims_tuple_str,
                "--agent.critic.activation",
                req.network.activation,
                "--agent.algorithm.learning-rate",
                str(req.ppo.learning_rate),
                "--agent.algorithm.gamma",
                str(req.ppo.gamma),
                "--agent.algorithm.entropy-coef",
                str(req.ppo.entropy_coef),
                "--agent.algorithm.clip-param",
                str(req.ppo.clip_param),
                "--agent.algorithm.value-loss-coef",
                str(req.ppo.value_loss_coef),
                "--agent.max-iterations",
                str(req.max_iterations),
                "--agent.run-name",
                run_name,
                "--agent.logger",
                "tensorboard",
            ]

            # Neutralize wandb network attempts & set unbuffered I/O
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["WANDB_MODE"] = "disabled"
            env["WANDB_DISABLED"] = "true"
            env["WANDB_SILENT"] = "true"

            # Launch detached subprocess
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            log_file = open(self.log_file_path, "w", encoding="utf-8", errors="replace")
            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(MICRODUCK_RL_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
            self.current_pid = self.current_process.pid

            # SQLite persistence
            db.create_train_run(
                run_name=run_name,
                task_id=env_id,
                config=req.model_dump(),
                pid=self.current_pid,
            )

            # Background stdout reader thread
            threading.Thread(
                target=self._reader_thread,
                args=(self.current_process, log_file, req.max_iterations),
                daemon=True,
            ).start()

            # Schedule async monitor task
            self._monitor_task = asyncio.create_task(self._monitor_process())

            return {
                "run_name": run_name,
                "task_id": env_id,
                "pid": self.current_pid,
                "status": "running",
            }

    def _reader_thread(self, proc: subprocess.Popen, log_file, max_iterations: int) -> None:
        """Reads subprocess stdout line by line."""
        log_dir_pattern = re.compile(r"Logging experiment in directory:\s*(.+)")
        ansi_pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
        iter_pattern = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
        reward_pattern = re.compile(r"Mean reward:\s*([\-\d\.]+)")
        eta_pattern = re.compile(r"ETA:\s*([\d:]+)")
        elapsed_pattern = re.compile(r"Time elapsed:\s*([\d:]+)")

        try:
            for line in iter(proc.stdout.readline, ""):
                raw_clean = line.rstrip()
                clean_line = ansi_pattern.sub("", raw_clean)
                self.log_lines.append(clean_line)
                print(f"[Train] {clean_line}", flush=True)
                log_file.write(line)
                log_file.flush()

                # Realtime progress extraction directly from stdout
                m_iter = iter_pattern.search(clean_line)
                if m_iter:
                    self.current_iter = int(m_iter.group(1))
                    self.max_iter = int(m_iter.group(2))
                    self.latest_log_summary = f"正在训练第 {self.current_iter}/{self.max_iter} 轮策略..."

                m_rew = reward_pattern.search(clean_line)
                if m_rew:
                    try:
                        self.mean_reward = float(m_rew.group(1))
                    except ValueError:
                        pass

                m_eta = eta_pattern.search(clean_line)
                if m_eta:
                    self.eta_str = m_eta.group(1)

                m_elap = elapsed_pattern.search(clean_line)
                if m_elap:
                    self.elapsed_str = m_elap.group(1)

                # 1. Detect log_dir from stdout
                if not self.current_log_dir:
                    match = log_dir_pattern.search(clean_line)
                    if match:
                        found_path = Path(match.group(1).strip())
                        if not found_path.is_absolute():
                            found_path = MICRODUCK_RL_DIR / found_path
                        self.current_log_dir = found_path
                        if self._loop and not self._loop.is_closed():
                            self._loop.call_soon_threadsafe(
                                self._start_metrics_tailer,
                                self.current_log_dir,
                                max_iterations,
                            )

                # 2. Fallback filesystem scan for rsl_rl output dir in case stdout line was buffered
                if not self.current_log_dir and self.current_run_name:
                    rsl_logs = MICRODUCK_RL_DIR / "logs" / "rsl_rl"
                    if rsl_logs.exists():
                        matching = list(rsl_logs.glob(f"**/*{self.current_run_name}*"))
                        if matching:
                            matching.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            self.current_log_dir = matching[0]
                            if self._loop and not self._loop.is_closed():
                                self._loop.call_soon_threadsafe(
                                    self._start_metrics_tailer,
                                    self.current_log_dir,
                                    max_iterations,
                                )
        except Exception as e:
            self.log_lines.append(f"[TrainingManager] Reader error: {e}")
        finally:
            log_file.close()

    async def _monitor_process(self) -> None:
        """Monitors when the subprocess exits and finalizes state."""
        proc = self.current_process
        run_name = self.current_run_name
        if not proc or not run_name:
            return

        exit_code = await asyncio.to_thread(proc.wait)
        if self.metrics_tailer:
            self.metrics_tailer.stop()

        if exit_code == 0:
            self.current_status = "finished"
        elif self.current_status == "stopped":
            pass
        else:
            self.current_status = "error"
            recent_err = [l for l in list(self.log_lines)[-15:] if l.strip() and not l.startswith("[INFO]")]
            self.current_error = " | ".join(recent_err[-3:]) if recent_err else f"Process exited with code {exit_code}"

        db.update_train_run_status(
            run_name=run_name,
            status=self.current_status,
            exit_code=exit_code,
            error_message=self.current_error,
            log_dir=str(self.current_log_dir) if self.current_log_dir else None,
        )

        if self.current_log_dir and self.current_log_dir.exists():
            for pt_file in self.current_log_dir.glob("model_*.pt"):
                iter_match = re.search(r"model_(\d+)\.pt", pt_file.name)
                iteration = int(iter_match.group(1)) if iter_match else None
                db.save_checkpoint(
                    run_name=run_name,
                    checkpoint_name=pt_file.name,
                    checkpoint_path=str(pt_file.resolve()),
                    iteration=iteration,
                )
            for onnx_file in self.current_log_dir.glob("*.onnx"):
                db.save_onnx_model(
                    name=f"{run_name}/{onnx_file.name}",
                    onnx_path=str(onnx_file.resolve()),
                    source_run_name=run_name,
                )

    async def stop_train(self) -> Dict[str, Any]:
        async with self._lock:
            if not self.is_running():
                return {"message": "当前没有正在运行的训练任务"}

            self.current_status = "stopped"
            pid = self.current_pid or (self.current_process.pid if self.current_process else None)
            if pid and psutil.pid_exists(pid):
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    parent.terminate()
                    gone, alive = psutil.wait_procs([parent], timeout=3)
                    for p in alive:
                        p.kill()
                except Exception as e:
                    return {"error": f"终止进程失败: {e}"}

            if self.metrics_tailer:
                try:
                    self.metrics_tailer.stop()
                except Exception:
                    pass
                self.metrics_tailer = None
            if self._tailer_task and not self._tailer_task.done():
                self._tailer_task.cancel()
                self._tailer_task = None

            return {"message": f"任务 '{self.current_run_name}' 已成功停止"}

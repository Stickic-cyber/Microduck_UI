"""Inference Manager: Coordinates offscreen rendering loop and WebSocket stream broadcasting."""

import asyncio
import threading
import time
from typing import Any, Dict, Optional, Set
from backend.policy_runner import PolicyRunner

class InferenceManager:
    """Manages PolicyRunner simulation thread and WebSocket frame broadcast."""

    def __init__(self):
        self.runner: Optional[PolicyRunner] = None
        self.active_onnx_path: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._subscribers: Set[asyncio.Queue] = set()
        self._seq = 0
        self.fps = 25
        self.jpeg_quality = 65
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def is_active(self) -> bool:
        return self._running and self.runner is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "active": self.is_active(),
            "onnx_path": self.active_onnx_path,
            "fps": self.fps,
            "paused": self.runner.is_paused if self.runner else False,
            "step": self.runner.step_count if self.runner else 0,
            "sim_time": round(self.runner.sim_time, 2) if self.runner else 0.0,
        }

    def start(self, onnx_path: str, fps: int = 25, jpeg_quality: int = 65) -> Dict[str, Any]:
        self.stop()

        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.active_onnx_path = onnx_path
        self.runner = PolicyRunner(onnx_path=onnx_path)

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self._running = True
        self._seq = 0
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

        return {
            "status": "started",
            "onnx_path": onnx_path,
            "fps": fps,
        }

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.runner:
            try:
                self.runner.close()
            except Exception:
                pass
        self.runner = None
        self.active_onnx_path = None

    def subscribe(self) -> asyncio.Queue:
        # Maxsize 2 allows immediate drop of older frames to prevent delay
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        if self.runner:
            self.runner.set_command(vx, vy, wz)

    def reset(self) -> None:
        if self.runner:
            self.runner.reset()

    def pause(self) -> None:
        if self.runner:
            self.runner.set_paused(True)

    def resume(self) -> None:
        if self.runner:
            self.runner.set_paused(False)

    def _broadcast_frame(self, frame_msg: Dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                if q.full():
                    # Frame dropping: drop oldest frame
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(frame_msg)
            except Exception:
                self._subscribers.discard(q)

    def _render_loop(self) -> None:
        dt = 1.0 / self.fps
        try:
            if self.runner:
                self.runner.init_renderer()

            while self._running and self.runner:
                t_start = time.time()
                try:
                    jpeg_b64, state_info = self.runner.step(quality=self.jpeg_quality)
                    self._seq += 1

                    frame_msg = {
                        "type": "frame",
                        "seq": self._seq,
                        "jpeg_b64": jpeg_b64,
                        "sim_time": state_info["sim_time"],
                        "step": state_info["step"],
                        "cmd": state_info["cmd"],
                    }

                    if self._loop and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(self._broadcast_frame, frame_msg)
                    else:
                        self._broadcast_frame(frame_msg)

                except Exception as e:
                    time.sleep(0.05)

                elapsed = time.time() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if self.runner:
                try:
                    self.runner.close()
                except Exception:
                    pass

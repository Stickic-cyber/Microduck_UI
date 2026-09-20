"""TensorBoard EventAccumulator reader for live streaming training metrics."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class MetricsTailer:
    """Monitors a training run directory and yields new scalar metrics."""

    def __init__(
        self,
        log_dir: Path,
        max_iterations: int = 1000,
        poll_interval: float = 0.8,
    ):
        self.log_dir = Path(log_dir)
        self.max_iterations = max_iterations
        self.poll_interval = poll_interval
        self._last_step = -1
        self._start_time = time.time()
        self._accumulator: Optional[EventAccumulator] = None
        self._running = False
        self._subscribers: Set[asyncio.Queue] = set()
        self._cached_metrics: List[Dict[str, Any]] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        # Replay latest points to newly connected client
        for m in self._cached_metrics[-100:]:
            q.put_nowait(m)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, data: Dict[str, Any]) -> None:
        self._cached_metrics.append(data)
        if len(self._cached_metrics) > 3000:
            self._cached_metrics = self._cached_metrics[-2000:]
        for q in list(self._subscribers):
            try:
                q.put_nowait(data)
            except Exception:
                self._subscribers.discard(q)

    def _find_event_file(self) -> Optional[Path]:
        if not self.log_dir.exists():
            return None
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            event_files = list(self.log_dir.glob("**/events.out.tfevents.*"))
        if event_files:
            event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return event_files[0]
        return None

    def read_latest_scalars(self) -> List[Dict[str, Any]]:
        event_file = self._find_event_file()
        if not event_file:
            return []

        if self._accumulator is None or self._accumulator.path != str(event_file):
            size_guidance = {
                "scalars": 10000,
                "tensors": 0,
                "images": 0,
                "audio": 0,
                "histograms": 0,
            }
            self._accumulator = EventAccumulator(str(event_file), size_guidance=size_guidance)

        try:
            self._accumulator.Reload()
        except Exception:
            return []

        scalar_tags = self._accumulator.Tags().get("scalars", [])
        if not scalar_tags:
            return []

        # Discovered actual tags from smoke test:
        # - Train/mean_reward
        # - Train/mean_episode_length
        # - Loss/value (or Loss/value_loss)
        # - Loss/surrogate
        # - Loss/entropy
        # - Perf/total_fps
        reward_tag = None
        for tag in ["Train/mean_reward", "Reward/mean_reward", "mean_reward", "Episode/reward"]:
            if tag in scalar_tags:
                reward_tag = tag
                break

        loss_tags = {}
        for tag in scalar_tags:
            tag_lower = tag.lower()
            if tag_lower.startswith("loss/"):
                short_key = tag[5:]  # strip 'Loss/'
                # Normalize 'value' to 'value_loss' for clarity
                if short_key == "value":
                    short_key = "value_loss"
                elif short_key == "surrogate":
                    short_key = "surrogate_loss"
                elif short_key == "entropy":
                    short_key = "entropy_loss"
                loss_tags[short_key] = tag

        fps_tag = None
        for tag in ["Perf/total_fps", "FPS", "Train/fps", "fps"]:
            if tag in scalar_tags:
                fps_tag = tag
                break

        # Collect events indexed by step
        step_data: Dict[int, Dict[str, Any]] = {}

        if reward_tag:
            for ev in self._accumulator.Scalars(reward_tag):
                step = int(ev.step)
                if step > self._last_step:
                    if step not in step_data:
                        step_data[step] = {"iteration": step, "losses": {}}
                    step_data[step]["mean_reward"] = float(ev.value)

        for lk, lt in loss_tags.items():
            for ev in self._accumulator.Scalars(lt):
                step = int(ev.step)
                if step > self._last_step:
                    if step not in step_data:
                        step_data[step] = {"iteration": step, "losses": {}}
                    step_data[step]["losses"][lk] = float(ev.value)

        if fps_tag:
            for ev in self._accumulator.Scalars(fps_tag):
                step = int(ev.step)
                if step in step_data:
                    step_data[step]["fps"] = float(ev.value)

        results = []
        for step in sorted(step_data.keys()):
            sd = step_data[step]
            mean_reward = sd.get("mean_reward", 0.0)
            losses = sd.get("losses", {})
            fps = sd.get("fps", 0.0)

            elapsed_s = time.time() - self._start_time
            if step > 0 and self.max_iterations > step:
                eta_s = (elapsed_s / step) * (self.max_iterations - step)
            else:
                eta_s = 0.0

            metric_msg = {
                "type": "metric",
                "iteration": step,
                "mean_reward": round(mean_reward, 3),
                "loss": {k: round(v, 4) for k, v in losses.items()},
                "fps": round(fps, 1),
                "elapsed_s": round(elapsed_s, 1),
                "eta_s": round(eta_s, 1),
            }
            results.append(metric_msg)
            self._last_step = max(self._last_step, step)

        return results

    async def run_loop(self) -> None:
        self._running = True
        while self._running:
            try:
                new_metrics = self.read_latest_scalars()
                for m in new_metrics:
                    self._broadcast(m)
            except Exception:
                pass
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False

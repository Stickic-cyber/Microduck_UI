"""FastAPI application entry point: REST API routes and WebSockets for Microduck RL WebUI."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import db
from backend.config import (
    EXPORTS_DIR,
    FRONTEND_DIST_DIR,
    MICRODUCK_RL_DIR,
    UV_BIN,
    get_available_onnx_models,
)
from backend.inference_manager import InferenceManager
from backend.schemas import (
    ExportRequest,
    InferCommand,
    InferStartRequest,
    TrainStartRequest,
)
from backend.training_manager import TrainingManager

app = FastAPI(
    title="Microduck RL WebUI",
    description="Training Monitoring and ONNX Inference Web Dashboard for Microduck RL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_mgr = TrainingManager()
inference_mgr = InferenceManager()

@app.on_event("startup")
def on_startup():
    db.init_db()

# --- REST APIs ---

@app.get("/api/envs")
def get_environments() -> List[str]:
    """Return registered RL environments."""
    try:
        import mjlab.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks
        tasks = list_tasks()
        # Filter and sort microduck tasks first
        duck_tasks = [t for t in tasks if "MicroDuck" in t]
        other_tasks = [t for t in tasks if "MicroDuck" not in t]
        return duck_tasks + other_tasks
    except Exception:
        # Fallback list if mjlab import fails
        return [
            "Mjlab-Velocity-Flat-MicroDuck",
            "Mjlab-Velocity-Rough-MicroDuck",
            "Mjlab-VelStand-Flat-MicroDuck",
            "Mjlab-StandUp-Flat-MicroDuck",
            "Mjlab-SitStand-Flat-MicroDuck",
            "Mjlab-GroundPick-Flat-MicroDuck",
            "Mjlab-Roulade-Flat-MicroDuck",
            "Mjlab-Velocity-Flat-MicroDuck-Rollers",
            "Mjlab-Velocity-Flat-Backlash-MicroDuck",
        ]

@app.get("/api/train/status")
def get_train_status() -> Dict[str, Any]:
    """Get current training status."""
    return training_mgr.get_status()

@app.post("/api/train/start")
async def start_train(req: TrainStartRequest) -> Dict[str, Any]:
    """Start a new training run."""
    try:
        res = await training_mgr.start_train(req)
        return res
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/train/stop")
async def stop_train() -> Dict[str, Any]:
    """Stop active training run."""
    return await training_mgr.stop_train()

@app.get("/api/train/history")
def get_train_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get historical training runs from SQLite."""
    return db.get_train_runs(limit=limit)

@app.get("/api/train/logs")
def get_train_logs(limit: int = 100) -> Dict[str, Any]:
    """Get latest stdout/stderr logs from the current run."""
    return {
        "run_name": training_mgr.current_run_name,
        "logs": training_mgr.get_logs(limit=limit),
    }

@app.get("/api/checkpoints")
def list_checkpoints(run_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available model checkpoints (.pt)."""
    checkpoints = db.get_checkpoints(run_name=run_name)
    if not checkpoints:
        # Also scan local filesystem ckpt directory
        ckpt_dir = MICRODUCK_RL_DIR / "ckpt"
        if ckpt_dir.exists():
            for p in ckpt_dir.glob("**/*.pt"):
                checkpoints.append({
                    "run_name": p.parent.name,
                    "checkpoint_name": p.name,
                    "checkpoint_path": str(p.resolve()),
                    "created_at": "",
                })
    return checkpoints

@app.post("/api/export")
async def export_onnx(req: ExportRequest) -> Dict[str, Any]:
    """Wrap scripts/export.py to convert a checkpoint (.pt) to ONNX."""
    checkpoint_path = Path(req.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (MICRODUCK_RL_DIR / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

    filename = req.onnx_filename or f"model_{checkpoint_path.stem}.onnx"
    if not filename.endswith(".onnx"):
        filename += ".onnx"
    output_onnx = EXPORTS_DIR / filename

    cmd = [
        UV_BIN,
        "run",
        "scripts/export.py",
        req.task_id,
        "--checkpoint-file",
        str(checkpoint_path),
        "--onnx-file",
        str(output_onnx),
        "--device",
        "cpu",
    ]

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=str(MICRODUCK_RL_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Export failed (code {proc.returncode}): {proc.stderr or proc.stdout}",
            )

        db.save_onnx_model(
            name=filename,
            onnx_path=str(output_onnx.resolve()),
            source_run_name=req.run_name,
            source_checkpoint=str(checkpoint_path),
        )

        return {
            "status": "success",
            "onnx_name": filename,
            "onnx_path": str(output_onnx.resolve()),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Export script timed out.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onnx/list")
def list_onnx_models() -> List[Dict[str, Any]]:
    """List all available ONNX models for inference."""
    return get_available_onnx_models()

@app.get("/api/infer/status")
def get_infer_status() -> Dict[str, Any]:
    """Get active inference status."""
    return inference_mgr.get_status()

@app.post("/api/infer/start")
def start_infer(req: InferStartRequest) -> Dict[str, Any]:
    """Start headless inference and video stream for given ONNX."""
    try:
        return inference_mgr.start(
            onnx_path=req.onnx_path,
            fps=req.fps,
            jpeg_quality=req.jpeg_quality,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/infer/stop")
def stop_infer() -> Dict[str, Any]:
    """Stop active inference."""
    inference_mgr.stop()
    return {"status": "stopped"}

@app.post("/api/infer/command")
def send_infer_command(cmd: InferCommand) -> Dict[str, Any]:
    """Send teleop velocity command or action (reset/pause/resume)."""
    if cmd.action == "reset":
        inference_mgr.reset()
    elif cmd.action == "pause":
        inference_mgr.pause()
    elif cmd.action == "resume":
        inference_mgr.resume()

    inference_mgr.set_command(cmd.vx, cmd.vy, cmd.wz)
    return {"status": "ok", "cmd": {"vx": cmd.vx, "vy": cmd.vy, "wz": cmd.wz}}

# --- WebSockets ---

@app.websocket("/ws/train_metrics")
async def ws_train_metrics(websocket: WebSocket):
    """Streams training scalar metrics and status updates to frontend."""
    await websocket.accept()
    # Send initial status
    status = training_mgr.get_status()
    await websocket.send_json({
        "type": "status",
        **status,
    })

    q = None
    current_tailer = None

    try:
        while True:
            # Dynamically switch subscription when a new training run starts
            if training_mgr.metrics_tailer != current_tailer:
                if q and current_tailer:
                    try:
                        current_tailer.unsubscribe(q)
                    except Exception:
                        pass
                current_tailer = training_mgr.metrics_tailer
                q = current_tailer.subscribe() if current_tailer else None

            if q:
                try:
                    metric_data = await asyncio.wait_for(q.get(), timeout=0.8)
                    await websocket.send_json(metric_data)
                except asyncio.TimeoutError:
                    # Periodically send rich status heartbeat
                    curr_status = training_mgr.get_status()
                    await websocket.send_json({
                        "type": "status",
                        **curr_status,
                    })
            else:
                await asyncio.sleep(0.8)
                curr_status = training_mgr.get_status()
                await websocket.send_json({
                    "type": "status",
                    **curr_status,
                })
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        if q and current_tailer:
            try:
                current_tailer.unsubscribe(q)
            except Exception:
                pass

@app.websocket("/ws/render_stream")
async def ws_render_stream(websocket: WebSocket):
    """Streams video frames to frontend and accepts bidirectional teleop commands."""
    await websocket.accept()
    q = inference_mgr.subscribe()

    async def receive_commands():
        try:
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")
                if msg_type == "command":
                    vx = float(msg.get("vx", 0.0))
                    vy = float(msg.get("vy", 0.0))
                    wz = float(msg.get("wz", 0.0))
                    inference_mgr.set_command(vx, vy, wz)
                elif msg_type == "action":
                    action = msg.get("action")
                    if action == "reset":
                        inference_mgr.reset()
                    elif action == "pause":
                        inference_mgr.pause()
                    elif action == "resume":
                        inference_mgr.resume()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    recv_task = asyncio.create_task(receive_commands())

    try:
        while True:
            frame_msg = await q.get()
            await websocket.send_json(frame_msg)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        recv_task.cancel()
        inference_mgr.unsubscribe(q)

# --- Frontend Static Files & SPA Fallback ---
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        target_file = FRONTEND_DIST_DIR / full_path
        if full_path and target_file.exists() and not target_file.is_dir():
            return FileResponse(str(target_file))
        return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))

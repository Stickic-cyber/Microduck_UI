"""Configuration settings and path resolution for Microduck RL WebUI."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent
MICRODUCK_RL_DIR = WORKSPACE_ROOT / "microduck_rl" if (WORKSPACE_ROOT / "microduck_rl").exists() else BASE_DIR.parent

EXPORTS_DIR = BASE_DIR / "exports"
RUNS_DIR = BASE_DIR / "runs"
DB_PATH = BASE_DIR / "webui.db"
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# MuJoCo scene XML
SCENE_XML_PATH = MICRODUCK_RL_DIR / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"
if not SCENE_XML_PATH.exists():
    alt_scene = WORKSPACE_ROOT / "scene.xml"
    if alt_scene.exists():
        SCENE_XML_PATH = alt_scene

def get_available_onnx_models() -> list[dict]:
    models = []
    seen = set()
    # 1. Search in exports/
    for p in sorted(EXPORTS_DIR.glob("*.onnx")):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            models.append({"name": p.name, "path": resolved, "source": "exports"})
    # 2. Search in microduck_rl root
    for p in sorted(MICRODUCK_RL_DIR.glob("*.onnx")):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            models.append({"name": p.name, "path": resolved, "source": "microduck_rl"})
    # 3. Search in microduck_rl/ckpt/
    ckpt_dir = MICRODUCK_RL_DIR / "ckpt"
    if ckpt_dir.exists():
        for p in sorted(ckpt_dir.glob("**/*.onnx")):
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                models.append({"name": f"ckpt/{p.name}", "path": resolved, "source": "ckpt"})
    # 4. Search in microduck_rl/logs/
    logs_dir = MICRODUCK_RL_DIR / "logs"
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("**/*.onnx")):
            resolved = str(p.resolve())
            if resolved not in seen:
                seen.add(resolved)
                models.append({"name": f"logs/{p.parent.name}/{p.name}", "path": resolved, "source": "training_run"})
    # 5. Search in workspace root
    for p in sorted(WORKSPACE_ROOT.glob("*.onnx")):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            models.append({"name": p.name, "path": resolved, "source": "workspace"})
    return models

# Python and uv executable paths
VENV_PYTHON = MICRODUCK_RL_DIR / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(os.environ.get("PYTHON_BIN", "python"))

UV_BIN = "uv"
HOST = "0.0.0.0"
PORT = 8000

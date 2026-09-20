"""Microduck RL WebUI Launcher Script."""

import os
import sys
import time
import webbrowser
import threading

# Windows UTF-8 stdout encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

def open_browser():
    time.sleep(1.5)
    url = "http://localhost:8000"
    print(f"\n[Microduck RL WebUI] Opening browser at {url} ...")
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    print("=" * 75)
    print("  Microduck RL WebUI -- Training & Teleop Dashboard")
    print("  MuJoCo Warp + RSL-RL PPO + Headless Offscreen Rendering")
    print("=" * 75)
    print("  Local Server: http://localhost:8000")
    print("  REST API Docs: http://localhost:8000/docs")
    print("=" * 75)

    threading.Thread(target=open_browser, daemon=True).start()

    # NOTE: workers=1 is intentionally enforced here to guarantee:
    # 1. TrainingManager process exclusivity (GPU memory and singleton subprocess locking)
    # 2. WebSocket in-memory connection and metric queue consistency
    # 3. Direct access to the active MuJoCo PolicyRunner simulation thread
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, workers=1, log_level="info")

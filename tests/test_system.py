"""Automated end-to-end integration test suite for Microduck RL WebUI."""

import os
import sys
import time
import unittest
from pathlib import Path

# Ensure microduck_rl_UI is in sys.path
UI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UI_ROOT))

from backend import config, db
from backend.policy_runner import PolicyRunner
from backend.schemas import TrainStartRequest, InferCommand
from fastapi.testclient import TestClient
from backend.app import app

class TestMicroduckWebUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_01_config_paths(self):
        self.assertTrue(config.MICRODUCK_RL_DIR.exists(), "MICRODUCK_RL_DIR must exist")
        self.assertTrue(config.SCENE_XML_PATH.exists(), f"scene.xml must exist: {config.SCENE_XML_PATH}")
        models = config.get_available_onnx_models()
        self.assertGreater(len(models), 0, "At least one ONNX model should be detected")
        print(f"[TEST 1 PASS] Detected {len(models)} ONNX models: {[m['name'] for m in models]}")

    def test_02_database_operations(self):
        run_name = f"test_run_verify_{int(time.time())}"
        run = db.create_train_run(
            run_name=run_name,
            task_id="Mjlab-Velocity-Flat-MicroDuck",
            config={"num_envs": 16, "max_iterations": 50},
            pid=12345,
        )
        self.assertIsNotNone(run)
        self.assertEqual(run["run_name"], run_name)

        db.update_train_run_status(run_name, "finished", exit_code=0)
        updated = db.get_train_run(run_name)
        self.assertEqual(updated["status"], "finished")

        db.save_checkpoint(run_name, "model_50.pt", "/fake/path/model_50.pt", iteration=50)
        ckpts = db.get_checkpoints(run_name)
        self.assertEqual(len(ckpts), 1)
        self.assertEqual(ckpts[0]["checkpoint_name"], "model_50.pt")
        print("[TEST 2 PASS] SQLite database CRUD verified.")

    def test_03_policy_runner(self):
        models = config.get_available_onnx_models()
        target_model = models[0]["path"]
        runner = PolicyRunner(onnx_path=target_model)
        self.assertIn(runner.obs_dim, (51, 61), f"Unexpected obs_dim: {runner.obs_dim}")

        # Run 5 steps
        for step_idx in range(5):
            runner.set_command(0.2, 0.0, 0.1)
            jpeg_b64, state_info = runner.step(quality=60)
            self.assertGreater(len(jpeg_b64), 1000)
            self.assertEqual(state_info["step"], step_idx + 1)

        runner.reset()
        self.assertEqual(runner.step_count, 0)
        print("[TEST 3 PASS] PolicyRunner simulation, ONNX inference, and offscreen render verified.")

    def test_04_fastapi_endpoints(self):
        client = TestClient(app)

        # 1. Test GET /api/envs
        res = client.get("/api/envs")
        self.assertEqual(res.status_code, 200)
        envs = res.json()
        self.assertIn("Mjlab-Velocity-Flat-MicroDuck", envs)

        # 2. Test GET /api/train/status
        res = client.get("/api/train/status")
        self.assertEqual(res.status_code, 200)
        self.assertIn("state", res.json())

        # 3. Test GET /api/onnx/list
        res = client.get("/api/onnx/list")
        self.assertEqual(res.status_code, 200)
        self.assertGreater(len(res.json()), 0)

        # 4. Test Infer endpoints
        models = res.json()
        target_model = models[0]["path"]

        res = client.post("/api/infer/start", json={"onnx_path": target_model, "fps": 20})
        self.assertEqual(res.status_code, 200)

        res = client.post("/api/infer/command", json={"vx": 0.2, "vy": -0.1, "wz": 0.5})
        self.assertEqual(res.status_code, 200)

        res = client.get("/api/infer/status")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["active"])

        res = client.post("/api/infer/stop")
        self.assertEqual(res.status_code, 200)

        # 5. Test Frontend SPA serving
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("Microduck RL WebUI", res.text)
        print("[TEST 4 PASS] FastAPI REST endpoints & SPA static files verified.")

if __name__ == "__main__":
    unittest.main()

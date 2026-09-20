"""SQLite database management for training runs, checkpoints, and ONNX models."""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS train_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT UNIQUE NOT NULL,
        task_id TEXT NOT NULL,
        status TEXT NOT NULL,
        config_json TEXT NOT NULL,
        log_dir TEXT,
        pid INTEGER,
        created_at TEXT NOT NULL,
        ended_at TEXT,
        exit_code INTEGER,
        error_message TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT NOT NULL,
        checkpoint_name TEXT NOT NULL,
        checkpoint_path TEXT NOT NULL,
        iteration INTEGER,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS onnx_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        onnx_path TEXT UNIQUE NOT NULL,
        source_run_name TEXT,
        source_checkpoint TEXT,
        created_at TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def create_train_run(
    run_name: str,
    task_id: str,
    config: Dict[str, Any],
    log_dir: Optional[str] = None,
    pid: Optional[int] = None,
) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT OR REPLACE INTO train_runs (run_name, task_id, status, config_json, log_dir, pid, created_at)
        VALUES (?, ?, 'running', ?, ?, ?, ?)
        """,
        (run_name, task_id, json.dumps(config), log_dir, pid, now),
    )
    conn.commit()
    conn.close()
    return get_train_run(run_name)  # type: ignore

def update_train_run_status(
    run_name: str,
    status: str,
    exit_code: Optional[int] = None,
    error_message: Optional[str] = None,
    ended_at: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    if ended_at is None and status in ("finished", "stopped", "error"):
        ended_at = datetime.now().isoformat()
    
    query = "UPDATE train_runs SET status = ?"
    params: List[Any] = [status]
    if exit_code is not None:
        query += ", exit_code = ?"
        params.append(exit_code)
    if error_message is not None:
        query += ", error_message = ?"
        params.append(error_message)
    if ended_at is not None:
        query += ", ended_at = ?"
        params.append(ended_at)
    if log_dir is not None:
        query += ", log_dir = ?"
        params.append(log_dir)
        
    query += " WHERE run_name = ?"
    params.append(run_name)
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def get_train_run(run_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM train_runs WHERE run_name = ?", (run_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["config"] = json.loads(d["config_json"])
        return d
    return None

def get_latest_train_run() -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM train_runs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["config"] = json.loads(d["config_json"])
        return d
    return None

def get_train_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM train_runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d["config_json"])
        result.append(d)
    return result

def save_checkpoint(
    run_name: str,
    checkpoint_name: str,
    checkpoint_path: str,
    iteration: Optional[int] = None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT INTO checkpoints (run_name, checkpoint_name, checkpoint_path, iteration, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_name, checkpoint_name, checkpoint_path, iteration, now),
    )
    conn.commit()
    conn.close()

def get_checkpoints(run_name: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    if run_name:
        cursor.execute("SELECT * FROM checkpoints WHERE run_name = ? ORDER BY id DESC", (run_name,))
    else:
        cursor.execute("SELECT * FROM checkpoints ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_onnx_model(
    name: str,
    onnx_path: str,
    source_run_name: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        """
        INSERT OR REPLACE INTO onnx_models (name, onnx_path, source_run_name, source_checkpoint, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, onnx_path, source_run_name, source_checkpoint, now),
    )
    conn.commit()
    conn.close()

def get_onnx_models_from_db() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM onnx_models ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

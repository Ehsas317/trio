"""Trio Web Dashboard - Flask-based monitoring interface."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from flask import Flask, jsonify, redirect, render_template, request, url_for

CORE_DIR = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(CORE_DIR.parent))

from core.controller import ModelController
from core.memory_manager import MemoryManager
from core.state_manager import StateManager
from core.vector_store import VectorStoreClient

CONFIG_FILE = Path(__file__).with_suffix(".json")
config: Dict[str, Any] = {}

app = Flask(__name__, template_folder="templates")
state_manager: Optional[StateManager] = None
memory_manager: Optional[MemoryManager] = None
model_controller: Optional[ModelController] = None
vector_store: Optional[VectorStoreClient] = None


def load_config() -> bool:
    global config
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        config["host"] = config.get("host", "0.0.0.0")
        config["port"] = int(config.get("port", 5001))
        config["refresh_interval_sec"] = int(config.get("refresh_interval_sec", 5))
        return True
    except Exception as e:
        logging.error(f"Dashboard config error: {e}")
        return False


def setup_logging() -> None:
    level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    app.logger.setLevel(level)
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "dashboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "dashboard.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    app.logger.addHandler(handler)


def get_status_data() -> Dict[str, Any]:
    data: Dict[str, Any] = {"system": {}, "assistants": {}, "error": None}
    try:
        mem = memory_manager.get_system_memory_info() if memory_manager else {"error": "not initialized"}
        pressure = memory_manager.get_memory_pressure_macos() if memory_manager else {"error": "not initialized"}
        cpu = psutil.cpu_percent(interval=0.1)
        data["system"] = {
            "ram_used_mb": mem.get("used_mb", "N/A"),
            "ram_total_mb": mem.get("total_mb", "N/A"),
            "ram_percent": mem.get("percent_used", "N/A"),
            "cpu_percent": cpu,
            "wired_mb": pressure.get("wired_mb", "N/A"),
            "active_mb": pressure.get("active_mb", "N/A"),
            "compressed_mb": pressure.get("compressed_mb", "N/A"),
        }
    except Exception as e:
        data["system"]["error"] = str(e)

    try:
        states = state_manager.get_all_states() if state_manager else {}
        for name, st in states.items():
            data["assistants"][name] = {
                "status": st.status,
                "task": st.current_task,
                "progress": f"{st.task_progress:.1%}",
                "checkpoint_id": st.checkpoint_id,
                "last_updated": st.last_updated,
            }
    except Exception as e:
        data["error"] = str(e)

    if vector_store and vector_store.is_ready():
        data["vector_store_count"] = vector_store.count()
    else:
        data["vector_store_count"] = 0

    return data


@app.route("/")
def index():
    return render_template("index.html", refresh_interval=config.get("refresh_interval_sec", 5))


@app.route("/api/status")
def api_status():
    return jsonify(get_status_data())


@app.route("/control", methods=["POST"])
def control_assistant():
    if not model_controller:
        return redirect(url_for("index"))
    name = request.form.get("assistant")
    action = request.form.get("action")
    if name and action in ("start", "stop"):
        model_controller.send_command({"action": action, "assistant": name})
        app.logger.info(f"Sent {action} to {name}")
    return redirect(url_for("index"))


def main() -> None:
    global state_manager, memory_manager, model_controller, vector_store
    if not load_config():
        sys.exit(1)
    setup_logging()
    app.logger.info("Dashboard starting...")

    state_manager = StateManager()
    memory_manager = MemoryManager()
    model_controller = ModelController()
    vector_store = VectorStoreClient()

    host = config.get("host", "0.0.0.0")
    port = config.get("port", 5001)
    app.logger.info(f"Dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()

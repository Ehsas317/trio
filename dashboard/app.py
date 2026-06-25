"""Trio Web Dashboard - Flask-based monitoring interface."""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

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
        # FIX BUG-TRIO-003: Default to localhost only (safer than 0.0.0.0)
        config["host"] = config.get("host", "127.0.0.1")
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


def check_auth(username: str, password: str) -> bool:
    """Verify dashboard credentials from config."""
    auth = config.get("auth", {})
    expected_user = auth.get("username", "admin")
    expected_pass = auth.get("password", "")
    if not expected_pass:
        # No password configured — allow any (backward compat for local-only setups)
        return True
    return username == expected_user and password == expected_pass


def authenticate() -> Response:
    """Send 401 response with WWW-Authenticate header."""
    return Response(
        "Authentication required. Set password in dashboard.json.\n",
        401,
        {"WWW-Authenticate": 'Basic realm="Trio Dashboard"'},
    )


def require_auth(f):
    """Decorator to require HTTP Basic Auth for endpoints."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # Skip auth if no password is configured (local-only mode)
        auth_config = config.get("auth", {})
        if not auth_config.get("password"):
            return f(*args, **kwargs)

        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


def get_status_data() -> Dict[str, Any]:
    data: Dict[str, Any] = {"system": {}, "assistants": {}, "error": None}
    try:
        mem = memory_manager.get_system_memory_info() if memory_manager else {"error": "not initialized"}
        pressure = memory_manager.get_memory_pressure_macos() if memory_manager else {"error": "not initialized"}
        # FIX BUG-TRIO-004: Use interval=None for non-blocking cpu_percent.
        # The first call with interval=None returns the value since last call
        # (0.0 on first call), so we call once at startup to prime it.
        cpu = psutil.cpu_percent(interval=None)
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
@require_auth
def index():
    return render_template("index.html", refresh_interval=config.get("refresh_interval_sec", 5))


@app.route("/api/status")
@require_auth
def api_status():
    return jsonify(get_status_data())


@app.route("/control", methods=["POST"])
@require_auth
def control_assistant():
    if not model_controller:
        return redirect(url_for("index"))
    name = request.form.get("assistant")
    action = request.form.get("action")
    valid_actions = {"start", "stop", "pause", "resume"}
    if name and action in valid_actions:
        model_controller.send_command({"action": action, "assistant": name})
        app.logger.info(f"Sent {action} to {name}")
    return redirect(url_for("index"))


def main() -> None:
    global state_manager, memory_manager, model_controller, vector_store
    if not load_config():
        sys.exit(1)
    setup_logging()
    app.logger.info("Dashboard starting...")

    # FIX BUG-TRIO-004: Prime cpu_percent so interval=None returns useful data
    psutil.cpu_percent(interval=None)

    state_manager = StateManager()
    memory_manager = MemoryManager()
    model_controller = ModelController()
    vector_store = VectorStoreClient()

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 5001)
    app.logger.info(f"Dashboard at http://{host}:{port}")
    # SECURITY NOTE: Default is now 127.0.0.1. To expose externally,
    # set "host": "0.0.0.0" in dashboard.json AND configure a password
    # in the "auth" section.
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()

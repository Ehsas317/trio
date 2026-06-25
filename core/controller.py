"""Model Controller - Orchestrates assistant process lifecycle."""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.checkpoint_manager import CheckpointManager
from core.memory_manager import MemoryManager
from core.state_manager import AssistantState, StateManager

PROJECT_DIR = Path.home() / "trio_project_m4"

ASSISTANT_SCRIPTS = {
    "nami": PROJECT_DIR / "assistants" / "nami" / "nami_main.py",
    "rush": PROJECT_DIR / "assistants" / "rush" / "rush_main.py",
    "vex": PROJECT_DIR / "assistants" / "vex" / "vex_main.py",
}

LOG_FILE = PROJECT_DIR / "logs" / "controller" / "controller.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)


class ModelController:
    """Manages the lifecycle (start, stop, pause, resume) of AI assistant processes."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("ModelController")
        self.state_manager = StateManager()
        self.memory_manager = MemoryManager()
        self.checkpoint_manager = CheckpointManager()
        self.assistant_processes: Dict[str, Optional[subprocess.Popen]] = {
            name: None for name in ASSISTANT_SCRIPTS
        }
        self.assistant_pids: Dict[str, Optional[int]] = {
            name: None for name in ASSISTANT_SCRIPTS
        }
        # FIX BUG-TRIO-002: Track open file descriptors for cleanup
        self._assistant_log_fds: Dict[str, tuple] = {
            name: (None, None) for name in ASSISTANT_SCRIPTS
        }
        self.command_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self.command_processor_thread = threading.Thread(
            target=self._process_commands, daemon=True
        )
        self.process_monitor_thread = threading.Thread(
            target=self._monitor_processes, daemon=True
        )
        self.logger.info("ModelController initialized.")

    def start(self) -> None:
        self.logger.info("Starting background threads...")
        self.command_processor_thread.start()
        self.process_monitor_thread.start()

    def stop(self) -> None:
        self.logger.info("Stopping ModelController...")
        self._stop_event.set()
        for name in self.assistant_processes:
            proc = self.assistant_processes.get(name)
            if proc and proc.poll() is None:
                self.command_queue.put({"action": "stop", "assistant": name, "force": False})
        try:
            self.command_queue.join()
        except Exception:
            pass
        self.command_processor_thread.join(timeout=5)
        self.process_monitor_thread.join(timeout=2)
        for name, proc in list(self.assistant_processes.items()):
            if proc and proc.poll() is None:
                self._terminate_process(name, force=True)
        self.logger.info("ModelController stopped.")

    def send_command(self, command: Dict[str, Any]) -> None:
        self.command_queue.put(command)

    def _process_commands(self) -> None:
        self.logger.info("Command processor started.")
        while not self._stop_event.is_set():
            try:
                cmd = self.command_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                action = cmd.get("action")
                name = cmd.get("assistant")
                if not name or name not in ASSISTANT_SCRIPTS:
                    self.logger.error(f"Invalid assistant in command: {cmd}")
                    self.command_queue.task_done()
                    continue
                if action == "start":
                    self._start_assistant(name)
                elif action == "stop":
                    self._stop_assistant(name, force=cmd.get("force", False))
                elif action == "pause":
                    self._signal_assistant(name, signal.SIGUSR1)
                    self.state_manager.update_status(name, status="paused")
                elif action == "resume":
                    self._signal_assistant(name, signal.SIGUSR2)
                    self.state_manager.update_status(name, status="active")
                else:
                    self.logger.warning(f"Unknown action '{action}' in command")
            except Exception as e:
                self.logger.error(f"Error processing command: {e}", exc_info=True)
            finally:
                try:
                    self.command_queue.task_done()
                except ValueError:
                    pass
        self.logger.info("Command processor stopped.")

    def _monitor_processes(self) -> None:
        self.logger.info("Process monitor started.")
        while not self._stop_event.wait(5):
            for name, proc in list(self.assistant_processes.items()):
                if proc:
                    rc = proc.poll()
                    if rc is not None:
                        self.logger.warning(
                            f"Assistant '{name}' (PID {self.assistant_pids.get(name)}) "
                            f"terminated unexpectedly with code {rc}."
                        )
                        self.assistant_processes[name] = None
                        self.assistant_pids[name] = None
                        current = self.state_manager.load_state(name)
                        if current.status != "idle":
                            self.state_manager.update_status(
                                name, status="error", task="Process terminated unexpectedly"
                            )
        self.logger.info("Process monitor stopped.")

    def _start_assistant(self, name: str) -> None:
        proc = self.assistant_processes.get(name)
        if proc and proc.poll() is None:
            self.logger.warning(f"Assistant '{name}' already running.")
            return
        script_path = ASSISTANT_SCRIPTS.get(name)
        if not script_path or not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            self.state_manager.update_status(name, status="error", task=f"Script not found: {script_path}")
            return
        venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
        python_executable = str(venv_python) if venv_python.exists() else "python3"
        last_state = self.state_manager.load_state(name)
        checkpoint_id = None
        if last_state.status == "paused" and last_state.checkpoint_id:
            checkpoint_id = last_state.checkpoint_id
            self.logger.info(f"Resuming '{name}' from checkpoint '{checkpoint_id}'.")
        env = os.environ.copy()
        if checkpoint_id:
            env["TRIO_CHECKPOINT_ID"] = checkpoint_id
        try:
            log_dir = PROJECT_DIR / "logs" / name
            log_dir.mkdir(parents=True, exist_ok=True)

            # FIX BUG-TRIO-002: Open log files explicitly so we can close them later.
            # Previously these were opened inline and leaked file descriptors.
            stdout_path = log_dir / f"{name}_stdout.log"
            stderr_path = log_dir / f"{name}_stderr.log"
            stdout_f = open(stdout_path, "a")
            stderr_f = open(stderr_path, "a")

            process = subprocess.Popen(
                [python_executable, str(script_path)],
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
            )
            # Store file descriptors for cleanup in _terminate_process
            self._assistant_log_fds[name] = (stdout_f, stderr_f)

            self.assistant_processes[name] = process
            self.assistant_pids[name] = process.pid
            self.logger.info(f"Assistant '{name}' started with PID {process.pid}.")
            if checkpoint_id:
                self.state_manager.update_status(
                    name, status="active", task="Resuming task",
                    progress=last_state.task_progress, checkpoint_id=checkpoint_id,
                )
            else:
                self.state_manager.update_status(name, status="active", task="Initializing")
        except Exception as e:
            self.logger.error(f"Failed to start '{name}': {e}", exc_info=True)
            # Clean up any opened file descriptors on failure
            self._close_log_fds(name)
            self.assistant_processes[name] = None
            self.assistant_pids[name] = None
            self.state_manager.update_status(name, status="error", task=f"Failed to start: {e}")

    def _stop_assistant(self, name: str, force: bool = False) -> None:
        proc = self.assistant_processes.get(name)
        pid = self.assistant_pids.get(name)
        if not proc or not pid or proc.poll() is not None:
            self.logger.warning(f"Assistant '{name}' not running.")
            if self.state_manager.load_state(name).status != "idle":
                self.state_manager.update_status(name, status="idle")
            self._cleanup_assistant(name)
            return
        self.logger.info(f"Stopping '{name}' (PID {pid}). Force={force}")
        self.state_manager.update_status(name, status="idle")
        self._terminate_process(name, force=force)

    def _terminate_process(self, name: str, force: bool = False) -> None:
        proc = self.assistant_processes.get(name)
        pid = self.assistant_pids.get(name)
        if not proc or not pid or proc.poll() is not None:
            self._cleanup_assistant(name)
            return
        try:
            if not force:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                    self.logger.info(f"Assistant '{name}' terminated gracefully.")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"'{name}' did not terminate gracefully. Forcing.")
                    force = True
            if force:
                proc.kill()
                proc.wait(timeout=5)
                self.logger.info(f"Assistant '{name}' forcefully terminated.")
        except Exception as e:
            self.logger.error(f"Error terminating '{name}': {e}")
        finally:
            # FIX BUG-TRIO-002: Always close the log file descriptors
            self._close_log_fds(name)
            self._cleanup_assistant(name)

    def _close_log_fds(self, name: str) -> None:
        """Close log file descriptors for an assistant."""
        stdout_f, stderr_f = self._assistant_log_fds.get(name, (None, None))
        try:
            if stdout_f:
                stdout_f.close()
        except Exception:
            pass
        try:
            if stderr_f:
                stderr_f.close()
        except Exception:
            pass
        self._assistant_log_fds[name] = (None, None)

    def _cleanup_assistant(self, name: str) -> None:
        """Clean up assistant process tracking state."""
        self.assistant_processes[name] = None
        self.assistant_pids[name] = None

    def _signal_assistant(self, name: str, sig: int) -> None:
        proc = self.assistant_processes.get(name)
        pid = self.assistant_pids.get(name)
        if proc and pid and proc.poll() is None:
            try:
                proc.send_signal(sig)
                self.logger.debug(f"Signal {sig} sent to '{name}'.")
            except Exception as e:
                self.logger.error(f"Failed to send signal {sig} to '{name}': {e}")
        else:
            self.logger.warning(f"Cannot send signal to '{name}': not running.")

"""State Manager - Tracks operational status of all AI assistants."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_DIR = Path.home() / "trio_project_m4" / "shared" / "state"
ASSISTANT_NAMES = ["nami", "rush", "vex"]


class AssistantState:
    """Represents the state of a single AI assistant."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.status: str = "idle"
        self.current_task: Optional[str] = None
        self.task_progress: float = 0.0
        self.checkpoint_id: Optional[str] = None
        self.last_updated: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "current_task": self.current_task,
            "task_progress": self.task_progress,
            "checkpoint_id": self.checkpoint_id,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssistantState:
        state = cls(data.get("name", "unknown"))
        state.status = data.get("status", "idle")
        state.current_task = data.get("current_task")
        state.task_progress = data.get("task_progress", 0.0)
        state.checkpoint_id = data.get("checkpoint_id")
        state.last_updated = data.get("last_updated", time.time())
        return state


class StateManager:
    """Manages the state of all AI assistants."""

    def __init__(
        self,
        state_directory: str | Path = STATE_DIR,
        assistant_names: List[str] | None = None,
    ) -> None:
        self.state_dir = Path(state_directory)
        self.assistant_names = assistant_names or ASSISTANT_NAMES
        self._ensure_state_dir_exists()

    def _ensure_state_dir_exists(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating state directory {self.state_dir}: {e}")

    def _get_state_file_path(self, assistant_name: str) -> Path:
        return self.state_dir / f"{assistant_name}_state.json"

    def load_state(self, assistant_name: str) -> AssistantState:
        if assistant_name not in self.assistant_names:
            raise ValueError(f"Unknown assistant name: {assistant_name}")
        state_file = self._get_state_file_path(assistant_name)
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                return AssistantState.from_dict(data)
            except (json.JSONDecodeError, IOError, TypeError) as e:
                print(f"Error loading state for {assistant_name}: {e}. Returning default.")
        return AssistantState(assistant_name)

    def save_state(self, state: AssistantState) -> None:
        if state.name not in self.assistant_names:
            raise ValueError(f"Unknown assistant: {state.name}")
        state.last_updated = time.time()
        try:
            self._get_state_file_path(state.name).write_text(
                json.dumps(state.to_dict(), indent=4), encoding="utf-8"
            )
        except IOError as e:
            print(f"Error saving state for {state.name}: {e}")

    def get_all_states(self) -> Dict[str, AssistantState]:
        return {name: self.load_state(name) for name in self.assistant_names}

    def update_status(
        self,
        assistant_name: str,
        status: str,
        task: Optional[str] = None,
        progress: float = 0.0,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        valid_statuses = {"idle", "active", "paused", "error"}
        if status not in valid_statuses:
            print(f"Warning: Invalid status '{status}' for {assistant_name}. Using 'error'.")
            status = "error"
        state = self.load_state(assistant_name)
        state.status = status
        if status == "active":
            state.current_task = task
            state.task_progress = progress
            state.checkpoint_id = checkpoint_id
        elif status in ("idle", "error"):
            state.current_task = None
            state.task_progress = 0.0
        self.save_state(state)

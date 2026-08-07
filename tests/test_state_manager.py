"""Tests for core state_manager module."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import pytest

from core.state_manager import AssistantState, StateManager

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestAssistantState:
    def test_default_state(self) -> None:
        state = AssistantState("test")
        assert state.name == "test"
        assert state.status == "idle"
        assert state.current_task is None
        assert state.task_progress == 0.0
        assert state.checkpoint_id is None

    def test_to_dict(self) -> None:
        state = AssistantState("test")
        state.status = "active"
        state.current_task = "processing"
        state.task_progress = 0.5
        state.checkpoint_id = "ckpt_123"
        data = state.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "active"
        assert data["current_task"] == "processing"
        assert data["task_progress"] == 0.5
        assert data["checkpoint_id"] == "ckpt_123"

    def test_from_dict(self) -> None:
        data = {
            "name": "test",
            "status": "paused",
            "current_task": "task1",
            "task_progress": 0.75,
            "checkpoint_id": "ckpt_456",
            "last_updated": time.time(),
        }
        state = AssistantState.from_dict(data)
        assert state.name == "test"
        assert state.status == "paused"
        assert state.current_task == "task1"
        assert state.task_progress == 0.75
        assert state.checkpoint_id == "ckpt_456"

    def test_from_dict_missing_fields(self) -> None:
        data = {"name": "test"}
        state = AssistantState.from_dict(data)
        assert state.name == "test"
        assert state.status == "idle"
        assert state.current_task is None
        assert state.task_progress == 0.0
        assert state.checkpoint_id is None


class TestStateManager:
    def test_init_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            assert state_dir.exists()

    def test_load_state_creates_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            state = mgr.load_state("test")
            assert state.name == "test"
            assert state.status == "idle"

    def test_save_and_load_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            state = AssistantState("test")
            state.status = "active"
            state.current_task = "processing"
            state.task_progress = 0.5
            mgr.save_state(state)

            loaded = mgr.load_state("test")
            assert loaded.name == "test"
            assert loaded.status == "active"
            assert loaded.current_task == "processing"
            assert loaded.task_progress == 0.5

    def test_update_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            mgr.update_status("test", "active", task="running", progress=0.25)
            state = mgr.load_state("test")
            assert state.status == "active"
            assert state.current_task == "running"
            assert state.task_progress == 0.25

    def test_update_status_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            mgr.update_status("test", "invalid_status")
            state = mgr.load_state("test")
            assert state.status == "error"

    def test_get_all_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["a", "b"])
            mgr.update_status("a", "active")
            mgr.update_status("b", "paused")
            states = mgr.get_all_states()
            assert len(states) == 2
            assert states["a"].status == "active"
            assert states["b"].status == "paused"

    def test_unknown_assistant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / "state"
            mgr = StateManager(state_directory=state_dir, assistant_names=["test"])
            with pytest.raises(ValueError):
                mgr.load_state("unknown")
            with pytest.raises(ValueError):
                mgr.update_status("unknown", "active")
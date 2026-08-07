"""Tests for core checkpoint_manager module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from core.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    def test_init_creates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])
            assert ckpt_dir.exists()
            assert (ckpt_dir / "test").exists()

    def test_save_and_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            test_data = {"key": "value", "count": 42, "nested": {"a": 1}}
            cid = mgr.save_checkpoint("test", "task1", test_data, {"meta": "info"})

            assert cid is not None
            assert "test_task1_" in cid

            loaded_data, metadata = mgr.load_checkpoint(cid)
            assert loaded_data == test_data
            assert metadata["assistant_name"] == "test"
            assert metadata["task_identifier"] == "task1"
            assert metadata["custom_metadata"]["meta"] == "info"

    def test_save_checkpoint_unknown_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])
            cid = mgr.save_checkpoint("unknown", "task1", {"data": "test"})
            assert cid is None

    def test_load_nonexistent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])
            result = mgr.load_checkpoint("nonexistent_ckpt")
            assert result is None

    def test_find_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            cid1 = mgr.save_checkpoint("test", "task1", {"version": 1})
            time.sleep(0.01)  # Ensure different timestamps
            cid2 = mgr.save_checkpoint("test", "task1", {"version": 2})

            latest = mgr.find_latest_checkpoint("test", "task1")
            assert latest == cid2

    def test_find_latest_checkpoint_no_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            cid1 = mgr.save_checkpoint("test", "taskA", {"v": 1})
            time.sleep(0.01)
            cid2 = mgr.save_checkpoint("test", "taskB", {"v": 2})

            latest = mgr.find_latest_checkpoint("test")
            assert latest == cid2

    def test_delete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            cid = mgr.save_checkpoint("test", "task1", {"data": "test"})
            assert cid is not None

            deleted = mgr.delete_checkpoint(cid)
            assert deleted is True

            result = mgr.load_checkpoint(cid)
            assert result is None

    def test_delete_nonexistent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])
            deleted = mgr.delete_checkpoint("nonexistent")
            assert deleted is False

    def test_cleanup_old_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            # Create old checkpoint by manually writing with old timestamp
            old_cid = "test_task1_1000000000_abc12345"
            old_path = ckpt_dir / "test" / f"{old_cid}.json"
            old_path.write_text(
                '{"checkpoint_id": "%s", "assistant_name": "test", "task_identifier": "task1", "timestamp": 1000000000, "data": {}, "custom_metadata": {}}'
                % old_cid,
                encoding="utf-8",
            )

            # Create new checkpoint
            new_cid = mgr.save_checkpoint("test", "task2", {"v": 2})

            deleted = mgr.cleanup_old_checkpoints(days=1)
            assert deleted == 1
            assert mgr.load_checkpoint(new_cid) is not None

    def test_json_serialization_complex_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            mgr = CheckpointManager(checkpoint_directory=ckpt_dir, assistant_names=["test"])

            # Test with various JSON-serializable types
            test_data = {
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "none": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            }
            cid = mgr.save_checkpoint("test", "task", test_data)
            loaded_data, _ = mgr.load_checkpoint(cid)
            assert loaded_data == test_data
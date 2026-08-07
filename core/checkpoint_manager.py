"""Checkpoint Manager - Saves and loads task checkpoints for assistants.

Uses JSON for serialization instead of pickle for security.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

CHECKPOINT_DIR = Path.home() / "trio_project_m4" / "checkpoints"
ASSISTANT_NAMES = ["nami", "rush", "vex"]


class CheckpointEncoder(json.JSONEncoder):
    """Custom JSON encoder for checkpoint data."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        if hasattr(obj, "isoformat"):  # datetime
            return obj.isoformat()
        return super().default(obj)


def _json_serializable(data: Any) -> Any:
    """Convert data to JSON-serializable format."""
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, (list, tuple)):
        return [_json_serializable(item) for item in data]
    if isinstance(data, dict):
        return {str(k): _json_serializable(v) for k, v in data.items()}
    if hasattr(data, "__dict__"):
        return _json_serializable(data.__dict__)
    if hasattr(data, "tolist"):  # numpy arrays
        return data.tolist()
    if hasattr(data, "isoformat"):  # datetime
        return data.isoformat()
    return str(data)


class CheckpointManager:
    """Manages saving and loading task checkpoints using JSON."""

    def __init__(
        self,
        checkpoint_directory: str | Path = CHECKPOINT_DIR,
        assistant_names: List[str] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_directory)
        self.assistant_names = assistant_names or ASSISTANT_NAMES
        self._ensure_dirs_exist()

    def _ensure_dirs_exist(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in self.assistant_names:
            (self.checkpoint_dir / name).mkdir(parents=True, exist_ok=True)

    def _generate_id(self, assistant_name: str, task_identifier: str) -> str:
        ts = int(time.time())
        suffix = uuid.uuid4().hex[:8]
        return f"{assistant_name}_{task_identifier}_{ts}_{suffix}"

    def _get_checkpoint_path(self, assistant_name: str, checkpoint_id: str) -> Path:
        return self.checkpoint_dir / assistant_name / f"{checkpoint_id}.json"

    def save_checkpoint(
        self,
        assistant_name: str,
        task_identifier: str,
        checkpoint_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if assistant_name not in self.assistant_names:
            return None
        cid = self._generate_id(assistant_name, task_identifier)
        ckpt_path = self._get_checkpoint_path(assistant_name, cid)

        full_data = {
            "checkpoint_id": cid,
            "assistant_name": assistant_name,
            "task_identifier": task_identifier,
            "timestamp": time.time(),
            "data": _json_serializable(checkpoint_data),
            "custom_metadata": metadata or {},
        }

        try:
            ckpt_path.write_text(json.dumps(full_data, indent=4, cls=CheckpointEncoder), encoding="utf-8")
            return cid
        except Exception as e:
            ckpt_path.unlink(missing_ok=True)
            return None

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        assistant_from_id = checkpoint_id.split("_")[0]
        dirs = [self.checkpoint_dir / assistant_from_id] if assistant_from_id in self.assistant_names else []
        if not dirs:
            dirs = [self.checkpoint_dir / n for n in self.assistant_names]

        ckpt_path = None
        for d in dirs:
            cp = d / f"{checkpoint_id}.json"
            if cp.exists():
                ckpt_path = cp
                break

        if not ckpt_path:
            return None

        try:
            full_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            data = full_data.get("data")
            metadata = {
                "checkpoint_id": full_data.get("checkpoint_id"),
                "assistant_name": full_data.get("assistant_name"),
                "task_identifier": full_data.get("task_identifier"),
                "timestamp": full_data.get("timestamp"),
                "custom_metadata": full_data.get("custom_metadata", {}),
            }
            return data, metadata
        except Exception:
            return None

    def find_latest_checkpoint(
        self, assistant_name: str, task_identifier: Optional[str] = None
    ) -> Optional[str]:
        if assistant_name not in self.assistant_names:
            return None
        adir = self.checkpoint_dir / assistant_name
        if not adir.exists():
            return None
        checkpoints = []
        for fp in adir.glob("*.json"):
            try:
                meta = json.loads(fp.read_text(encoding="utf-8"))
                if task_identifier is None or meta.get("task_identifier") == task_identifier:
                    checkpoints.append(meta)
            except (IOError, json.JSONDecodeError):
                continue
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return checkpoints[0].get("checkpoint_id")

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        assistant_from_id = checkpoint_id.split("_")[0]
        dirs = [self.checkpoint_dir / assistant_from_id] if assistant_from_id in self.assistant_names else []
        if not dirs:
            dirs = [self.checkpoint_dir / n for n in self.assistant_names]
        deleted = False
        for d in dirs:
            cp = d / f"{checkpoint_id}.json"
            if cp.exists():
                cp.unlink(missing_ok=True)
                deleted = True
                break
        return deleted

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        cutoff = time.time() - (days * 86400)
        deleted = 0
        for name in self.assistant_names:
            adir = self.checkpoint_dir / name
            if not adir.exists():
                continue
            for fp in adir.glob("*.json"):
                try:
                    meta = json.loads(fp.read_text(encoding="utf-8"))
                    if meta.get("timestamp", 0) < cutoff:
                        cid = meta.get("checkpoint_id")
                        if cid and self.delete_checkpoint(cid):
                            deleted += 1
                except (IOError, json.JSONDecodeError):
                    continue
        return deleted
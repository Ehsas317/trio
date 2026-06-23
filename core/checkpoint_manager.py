"""Checkpoint Manager - Saves and loads task checkpoints for assistants."""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHECKPOINT_DIR = Path.home() / "trio_project_m4" / "checkpoints"
ASSISTANT_NAMES = ["nami", "rush", "vex"]


class CheckpointManager:
    """Manages saving and loading task checkpoints."""

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
        adir = self.checkpoint_dir / assistant_name
        pkl_path = adir / f"{cid}.pkl"
        meta_path = adir / f"{cid}.json"
        full_metadata = {
            "checkpoint_id": cid,
            "assistant_name": assistant_name,
            "task_identifier": task_identifier,
            "timestamp": time.time(),
            "pickle_file": pkl_path.name,
            "custom_data": metadata or {},
        }
        try:
            pkl_path.write_bytes(pickle.dumps(checkpoint_data))
            meta_path.write_text(json.dumps(full_metadata, indent=4), encoding="utf-8")
            return cid
        except Exception:
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        assistant_from_id = checkpoint_id.split("_")[0]
        dirs = [self.checkpoint_dir / assistant_from_id] if assistant_from_id in self.assistant_names else []
        if not dirs:
            dirs = [self.checkpoint_dir / n for n in self.assistant_names]
        meta_path = None
        for d in dirs:
            mp = d / f"{checkpoint_id}.json"
            if mp.exists():
                meta_path = mp
                break
        if not meta_path:
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            pkl_name = metadata.get("pickle_file", f"{checkpoint_id}.pkl")
            pkl_path = meta_path.parent / pkl_name
            if not pkl_path.exists():
                return None
            data = pickle.loads(pkl_path.read_bytes())
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
            mp = d / f"{checkpoint_id}.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                    pkl_name = meta.get("pickle_file")
                    if pkl_name:
                        (d / pkl_name).unlink(missing_ok=True)
                except Exception:
                    pass
                pp = d / f"{checkpoint_id}.pkl"
                mp.unlink(missing_ok=True)
                pp.unlink(missing_ok=True)
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

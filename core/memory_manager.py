"""Memory Manager - Monitors system memory usage and pressure."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional

import psutil


class MemoryManager:
    """Monitors system memory usage and pressure on macOS."""

    def get_system_memory_info(self) -> Dict[str, Any]:
        try:
            mem = psutil.virtual_memory()
            MB = 1024 * 1024
            return {
                "total_mb": mem.total // MB,
                "available_mb": mem.available // MB,
                "used_mb": mem.used // MB,
                "free_mb": mem.free // MB,
                "percent_used": mem.percent,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_memory_pressure_macos(self) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            stats: Dict[str, int] = {}
            for line in lines:
                if ":" in line:
                    key, value_str = line.split(":", 1)
                    key = key.strip()
                    value_str = value_str.strip().replace(".", "")
                    if value_str.isdigit():
                        stats[key] = int(value_str)

            pagesize = os.sysconf("SC_PAGESIZE")
            MB = 1024 * 1024

            return {
                "pressure_status": "unknown",
                "wired_mb": stats.get("Pages wired down", 0) * pagesize // MB,
                "active_mb": stats.get("Pages active", 0) * pagesize // MB,
                "inactive_mb": stats.get("Pages inactive", 0) * pagesize // MB,
                "free_mb": stats.get("Pages free", 0) * pagesize // MB,
                "compressed_mb": stats.get("Pages occupied by compressor", 0) * pagesize // MB,
            }
        except Exception as e:
            return {"error": str(e), "pressure_status": "error"}

    def get_process_memory_info(self, pid: int) -> Optional[Dict[str, Any]]:
        try:
            if not psutil.pid_exists(pid):
                return None
            proc = psutil.Process(pid)
            mem_info = proc.memory_info()
            mem_percent = proc.memory_percent()
            MB = 1024 * 1024
            return {
                "pid": pid,
                "name": proc.name(),
                "rss_mb": mem_info.rss // MB,
                "vms_mb": mem_info.vms // MB,
                "percent_used": round(mem_percent, 2),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            return None

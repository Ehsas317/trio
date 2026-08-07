"""Tests for core memory_manager module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.memory_manager import MemoryManager


class TestMemoryManager:
    def test_get_system_memory_info(self) -> None:
        mem = MemoryManager()
        info = mem.get_system_memory_info()
        assert "total_mb" in info
        assert "available_mb" in info
        assert "used_mb" in info
        assert "free_mb" in info
        assert "percent_used" in info
        assert isinstance(info["total_mb"], int)
        assert info["total_mb"] > 0

    @patch("psutil.virtual_memory")
    def test_get_system_memory_info_error(self, mock_vmem: MagicMock) -> None:
        mock_vmem.side_effect = Exception("test error")
        mem = MemoryManager()
        info = mem.get_system_memory_info()
        assert "error" in info

    @patch("subprocess.run")
    def test_get_memory_pressure_macos(self, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.stdout = "Pages free: 1000\nPages active: 2000\nPages inactive: 3000\nPages wired down: 4000\nPages occupied by compressor: 500\n"
        mock_run.return_value = mock_result

        with patch("os.sysconf", return_value=4096):
            mem = MemoryManager()
            pressure = mem.get_memory_pressure_macos()
            assert "wired_mb" in pressure
            assert "active_mb" in pressure
            assert "inactive_mb" in pressure
            assert "free_mb" in pressure
            assert "compressed_mb" in pressure
            assert pressure["pressure_status"] == "unknown"

    @patch("subprocess.run")
    def test_get_memory_pressure_macos_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = Exception("test error")
        mem = MemoryManager()
        pressure = mem.get_memory_pressure_macos()
        assert "error" in pressure
        assert pressure["pressure_status"] == "error"

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_get_process_memory_info(self, mock_process_class: MagicMock, mock_pid_exists: MagicMock) -> None:
        mock_pid_exists.return_value = True
        mock_proc = MagicMock()
        mock_proc.name.return_value = "test_process"
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_mem_info.vms = 1024 * 1024 * 200  # 200 MB
        mock_proc.memory_info.return_value = mock_mem_info
        mock_proc.memory_percent.return_value = 5.0
        mock_process_class.return_value = mock_proc

        mem = MemoryManager()
        info = mem.get_process_memory_info(1234)
        assert info is not None
        assert info["pid"] == 1234
        assert info["name"] == "test_process"
        assert info["rss_mb"] == 100
        assert info["vms_mb"] == 200
        assert info["percent_used"] == 5.0

    @patch("psutil.pid_exists")
    def test_get_process_memory_info_not_exists(self, mock_pid_exists: MagicMock) -> None:
        mock_pid_exists.return_value = False
        mem = MemoryManager()
        info = mem.get_process_memory_info(9999)
        assert info is None

    @patch("psutil.pid_exists")
    @patch("psutil.Process")
    def test_get_process_memory_info_access_denied(self, mock_process_class: MagicMock, mock_pid_exists: MagicMock) -> None:
        mock_pid_exists.return_value = True
        mock_process_class.side_effect = PermissionError("access denied")
        mem = MemoryManager()
        info = mem.get_process_memory_info(1234)
        assert info is None
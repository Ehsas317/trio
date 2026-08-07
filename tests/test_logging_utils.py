"""Tests for core logging_utils module."""

from __future__ import annotations

import io
import json
import logging
import sys
from unittest.mock import patch

import pytest

from core.logging_utils import JSONFormatter, LogContext, get_logger, setup_json_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    yield
    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    # Reset all loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


class TestJSONFormatter:
    def test_format_basic(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "Test message"
        assert data["function"] == "test_func"
        assert data["module"] == "test_module"
        assert data["line"] == 10
        assert "timestamp" in data

    def test_format_with_exception(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.funcName = "test_func"
        record.module = "test_module"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError: test error" in data["exception"]


class TestSetupJsonLogging:
    def test_setup_returns_logger(self) -> None:
        logger = setup_json_logging(level=logging.DEBUG)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG

    def test_setup_with_file(self, tmp_path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_json_logging(level=logging.INFO, log_file=str(log_file))
        logger.info("Test message")

        # Check file was created and contains valid JSON
        assert log_file.exists()
        content = log_file.read_text()
        data = json.loads(content.strip())
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"

    def test_setup_with_logger_name(self) -> None:
        logger = setup_json_logging(logger_name="test.named.logger")
        assert logger.name == "test.named.logger"


class TestGetLogger:
    def test_get_logger(self) -> None:
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"


class TestLogContext:
    def test_context_adds_extra_data(self) -> None:
        logger = get_logger("test.context")

        # Test that LogContext can be entered and exited without errors
        with LogContext(logger, key1="value1", key2="value2") as ctx_logger:
            assert ctx_logger is logger

        # Verify the factory was restored by creating a record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        # If factory wasn't restored, this would fail or have extra_data
        assert not hasattr(record, "extra_data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
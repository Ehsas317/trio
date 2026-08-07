"""Structured JSON logging utilities for the Trio AI Ecosystem."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def setup_json_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure structured JSON logging.

    Args:
        level: Logging level
        log_file: Optional file path for log output
        logger_name: Specific logger name, or None for root logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = JSONFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to log messages."""

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> logging.Logger:
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self.old_factory(*args, **kwargs)
            record.extra_data = self.context
            return record

        logging.setLogRecordFactory(record_factory)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logging.setLogRecordFactory(self.old_factory)
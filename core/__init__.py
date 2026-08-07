"""Core framework for the Trio AI Ecosystem."""

from core.checkpoint_manager import CheckpointManager
from core.controller import ModelController
from core.logging_utils import JSONFormatter, LogContext, get_logger, setup_json_logging
from core.memory_manager import MemoryManager
from core.state_manager import AssistantState, StateManager
from core.vector_store import VectorStoreClient

__all__ = [
    "CheckpointManager",
    "ModelController",
    "MemoryManager",
    "AssistantState",
    "StateManager",
    "VectorStoreClient",
    "JSONFormatter",
    "LogContext",
    "get_logger",
    "setup_json_logging",
]

__version__ = "2.0.0"
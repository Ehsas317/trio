"""Tests for core vector_store module - basic tests without heavy mocking."""

from __future__ import annotations

import pytest


class TestVectorStoreClient:
    def test_import_works(self) -> None:
        """Verify the module can be imported."""
        from core.vector_store import VectorStoreClient

        assert VectorStoreClient is not None

    def test_default_constants(self) -> None:
        """Test default constant values."""
        from core.vector_store import DEFAULT_MODEL, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION

        assert DEFAULT_MODEL == "all-MiniLM-L6-v2"
        assert DEFAULT_COLLECTION == "trio_knowledge_base"
        assert "trio_project_m4" in str(DEFAULT_PERSIST_DIR)

    def test_init_without_dependencies(self) -> None:
        """Test that VectorStoreClient initializes without errors."""
        from core.vector_store import VectorStoreClient

        # This should not raise an exception
        vs = VectorStoreClient()
        # Dependencies are installed, so it should be ready
        assert vs.is_ready() is True
        assert vs.model is not None
        assert vs.client is not None
        assert vs.collection is not None
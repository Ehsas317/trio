"""Vector Store - Cross-assistant knowledge sharing via ChromaDB."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = Path.home() / "trio_project_m4" / "vector_store"
DEFAULT_COLLECTION = "trio_knowledge_base"

logger = logging.getLogger("VectorStoreClient")


class VectorStoreClient:
    """Handles embedding generation and ChromaDB vector store interaction."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.model_name = model_name
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            self.model = None

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' ready.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            self.client = None
            self.collection = None

    def is_ready(self) -> bool:
        return self.model is not None and self.collection is not None

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_ready():
            logger.error("VectorStoreClient not ready.")
            return None
        if not text:
            logger.warning("Empty document text, skipping.")
            return None
        try:
            embedding = self.model.encode(text, convert_to_tensor=False).tolist()
            cid = doc_id or str(uuid.uuid4())
            cleaned = self._clean_metadata(metadata)
            self.collection.add(
                ids=[cid],
                embeddings=[embedding],
                documents=[text],
                metadatas=[cleaned],
            )
            logger.info(f"Document added: {cid}")
            return cid
        except Exception as e:
            logger.error(f"Failed to add document: {e}", exc_info=True)
            return None

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.is_ready():
            logger.error("VectorStoreClient not ready.")
            return None
        try:
            q_embed = self.model.encode(query_text, convert_to_tensor=False).tolist()
            results = self.collection.query(
                query_embeddings=[q_embed],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            formatted: List[Dict[str, Any]] = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                dists = results["distances"][0]
                for i in range(len(ids)):
                    formatted.append({
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i],
                        "distance": dists[i],
                    })
            logger.info(f"Query returned {len(formatted)} results.")
            return formatted
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return None

    def delete_document(self, doc_id: str) -> bool:
        if not self.is_ready():
            return False
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def count(self) -> int:
        if not self.is_ready():
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

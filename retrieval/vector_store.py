import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import Chunk, SourceMetadata


# Singleton QdrantClient instance cache to avoid local filesystem lock collision
_GLOBAL_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}


class QdrantVectorStore:
    """
    Production-grade wrapper for Qdrant Vector Database.
    Supports both local embedded storage (zero external daemon required) and
    remote Docker-hosted Qdrant server instances with Cosine distance metric.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: int = 384,
        client: Optional[QdrantClient] = None,
        storage_path: Optional[str] = None
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.dimension = dimension
        
        if client is not None:
            self.client = client
        elif settings.QDRANT_MODE == "remote" and settings.QDRANT_URL:
            logger.info(f"Connecting to remote Qdrant at {settings.QDRANT_URL}...")
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=5.0
            )
        else:
            # Embedded local storage or in-memory
            target_path = storage_path or settings.QDRANT_PATH
            if target_path == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                abs_path = os.path.abspath(target_path)
                os.makedirs(abs_path, exist_ok=True)
                if abs_path not in _GLOBAL_QDRANT_CLIENTS:
                    logger.info(f"Initializing embedded Qdrant database at {abs_path}...")
                    _GLOBAL_QDRANT_CLIENTS[abs_path] = QdrantClient(path=abs_path)
                self.client = _GLOBAL_QDRANT_CLIENTS[abs_path]

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists with Cosine metric and optimized HNSW config."""
        try:
            collections_resp = self.client.get_collections()
            existing_names = [c.name for c in collections_resp.collections]
            
            if self.collection_name not in existing_names:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with dimension {self.dimension}...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.dimension,
                        distance=rest_models.Distance.COSINE
                    ),
                    hnsw_config=rest_models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=1000
                    )
                )
        except Exception as e:
            logger.warning(f"Error checking/creating collection '{self.collection_name}': {e}")

    def upsert_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> bool:
        """Upsert a batch of chunks and their embedding vectors into Qdrant."""
        if not chunks or len(chunks) == 0:
            return True

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})")

        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vector = emb.tolist() if isinstance(emb, np.ndarray) else emb
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_document_id": chunk.parent_document_id,
                "text": chunk.text,
                "source": chunk.source,
                "language": chunk.language,
                "title": chunk.title,
                "dataset_split": chunk.dataset_split,
                "chunking_strategy": chunk.chunking_strategy,
                "chunk_position": chunk.chunk_position,
                "token_count": chunk.token_count,
                "character_count": chunk.character_count,
                "is_ground_truth": chunk.is_ground_truth,
                "metadata": chunk.metadata
            }
            point_id = abs(hash(chunk.chunk_id)) % (2**63 - 1)
            points.append(rest_models.PointStruct(id=point_id, vector=vector, payload=payload))

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            return True
        except Exception as e:
            logger.error(f"Failed upserting {len(points)} points to Qdrant: {e}")
            return False

    def search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        strategy_filter: Optional[str] = None,
        language_filter: Optional[str] = None
    ) -> List[SourceMetadata]:
        """
        Perform vector cosine similarity search in Qdrant with optional metadata filtering.
        """
        k = top_k or settings.TOP_K
        threshold = min_score if min_score is not None else settings.MIN_RETRIEVAL_SCORE
        vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        # Construct query filters
        must_conditions = []
        if strategy_filter:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="chunking_strategy",
                    match=rest_models.MatchValue(value=strategy_filter)
                )
            )
        if language_filter:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="language",
                    match=rest_models.MatchValue(value=language_filter)
                )
            )

        query_filter = rest_models.Filter(must=must_conditions) if must_conditions else None

        try:
            if hasattr(self.client, "query_points"):
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector_list,
                    limit=k,
                    score_threshold=threshold,
                    query_filter=query_filter
                ).points
            else:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector_list,
                    limit=k,
                    score_threshold=threshold,
                    query_filter=query_filter
                )

            sources: List[SourceMetadata] = []
            for hit in search_result:
                payload = hit.payload or {}
                sources.append(
                    SourceMetadata(
                        source_id=str(payload.get("chunk_id", hit.id)),
                        document_id=str(payload.get("document_id", "")),
                        relevance_score=float(hit.score),
                        chunking_strategy=str(payload.get("chunking_strategy", "unknown")),
                        text_excerpt=str(payload.get("text", "")),
                        title=str(payload.get("title", "")),
                        language=str(payload.get("language", "en")),
                        is_ground_truth=payload.get("is_ground_truth", False)
                    )
                )
            return sources
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def count(self) -> int:
        """Return total indexed vectors in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Check status of Qdrant collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status_str": str(info.status)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "collection": self.collection_name,
                "error": str(e)
            }

    def clear(self) -> None:
        """Wipe collection and recreate cleanly."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Cleared Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

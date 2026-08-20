from typing import List, Optional, Dict
import numpy as np
from pipeline.schemas import SourceMetadata
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever
from embeddings.base import BaseEmbedder
from config.settings import settings
from config.logging_config import logger


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Vector Similarity Search (Qdrant)
    and Lexical Search (BM25) using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_retriever: BM25LexicalRetriever,
        embedder: BaseEmbedder,
        dense_weight: float = 0.7,
        lexical_weight: float = 0.3,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        strategy_filter: Optional[str] = None,
        hybrid_enabled: Optional[bool] = None
    ) -> List[SourceMetadata]:
        """Execute hybrid or dense-only retrieval with reciprocal rank fusion."""
        k = top_k or settings.TOP_K
        use_hybrid = hybrid_enabled if hybrid_enabled is not None else settings.HYBRID_SEARCH

        # 1. Dense retrieval
        query_vector = self.embedder.embed_text(query)
        dense_candidates = self.vector_store.search(
            query_vector=query_vector,
            top_k=k * 2,
            min_score=min_score,
            strategy_filter=strategy_filter
        )

        if not use_hybrid:
            return dense_candidates[:k]

        # 2. Lexical retrieval
        bm25_candidates = self.bm25_retriever.search(
            query=query,
            top_k=k * 2,
            strategy_filter=strategy_filter
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        sources_by_id: Dict[str, SourceMetadata] = {}

        # Dense rank scoring
        for rank, item in enumerate(dense_candidates):
            sid = item.source_id
            sources_by_id[sid] = item
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)

        # BM25 rank scoring
        for rank, item in enumerate(bm25_candidates):
            sid = item.source_id
            if sid not in sources_by_id:
                sources_by_id[sid] = item
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + self.lexical_weight / (self.rrf_k + rank + 1)

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results: List[SourceMetadata] = []
        for sid in sorted_ids[:k]:
            source = sources_by_id[sid]
            # Update score with fused confidence
            source.relevance_score = round(rrf_scores[sid] * (self.rrf_k + 1), 4)
            results.append(source)

        return results

from typing import List, Optional
import numpy as np
from pipeline.schemas import SourceMetadata
from config.settings import settings
from config.logging_config import logger


class CrossEncoderReranker:
    """
    Reranks candidate retrieved chunks using a cross-attention model or fast feature reranker.
    Features a configurable fast-path bypass to maintain sub-200ms processing budgets.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: bool = False
    ):
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.enabled = enabled
        self._model = None

    def _init_model(self) -> None:
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder {self.model_name}: {e}. Reranker will run in fast heuristic mode.")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[SourceMetadata],
        top_k: int = 5,
        force_fast_path: bool = False
    ) -> List[SourceMetadata]:
        """Rerank candidates and return top_k most relevant chunks."""
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        # Fast path bypass (skips expensive neural scoring when latency target demands it)
        if not self.enabled or force_fast_path:
            return candidates[:top_k]

        self._init_model()

        if self._model is not None:
            try:
                pairs = [[query, c.text_excerpt] for c in candidates]
                scores = self._model.predict(pairs)
                
                # Pair candidates with new cross-encoder scores
                scored_candidates = []
                for candidate, score in zip(candidates, scores):
                    c_copy = candidate.model_copy()
                    c_copy.relevance_score = float(score)
                    scored_candidates.append(c_copy)
                    
                scored_candidates.sort(key=lambda x: x.relevance_score, reverse=True)
                return scored_candidates[:top_k]
            except Exception as e:
                logger.error(f"CrossEncoder inference failed: {e}. Returning original candidates.")
                return candidates[:top_k]

        # Fallback heuristic: score by query term overlap density + original score
        q_tokens = set(query.lower().split())
        for c in candidates:
            c_tokens = set(c.text_excerpt.lower().split())
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            c.relevance_score = round(c.relevance_score * 0.6 + overlap * 0.4, 4)
            
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        return candidates[:top_k]

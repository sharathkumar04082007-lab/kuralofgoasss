from typing import List, Optional
import numpy as np
import torch
from config.settings import settings
from config.logging_config import logger
from embeddings.base import BaseEmbedder
from embeddings.cache import EmbeddingCache


class MultilingualEmbedder(BaseEmbedder):
    """
    Multilingual embedding engine leveraging SentenceTransformers with
    deterministic caching, CPU inference optimizations, and unit normalization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_enabled: bool = True
    ):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_enabled = cache_enabled
        self.cache = EmbeddingCache(max_size=settings.EMBEDDING_CACHE_SIZE) if cache_enabled else None
        
        self._model = None
        self._dimension = 384  # Default for paraphrase-multilingual-MiniLM-L12-v2
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Lazy loads or initializes the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name} on {self.device}...")
            self._model = SentenceTransformer(self._model_name, device=self.device)
            # Warm up dimension
            sample_emb = self._model.encode("warmup", convert_to_numpy=True, normalize_embeddings=True)
            self._dimension = int(sample_emb.shape[0])
            logger.info(f"Embedding model ready. Dimension: {self._dimension}")
        except Exception as e:
            logger.warning(f"Failed loading SentenceTransformer {self._model_name}: {e}. Initializing deterministic fallback embedder.")
            self._model = None
            self._dimension = 384

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _deterministic_fallback_embed(self, text: str) -> np.ndarray:
        """Fast, deterministic hash-based 384-d pseudo embedding for offline/fallback test runs."""
        import hashlib
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized numpy vector."""
        text_clean = text.strip()
        if not text_clean:
            return np.zeros(self._dimension, dtype=np.float32)

        # Check Cache
        if self.cache:
            cached_vec = self.cache.get(text_clean, self._model_name)
            if cached_vec is not None:
                return cached_vec

        if self._model is not None:
            try:
                emb = self._model.encode(
                    text_clean,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                vec = np.array(emb, dtype=np.float32)
            except Exception as e:
                logger.error(f"Error during model encoding: {e}. Using fallback.")
                vec = self._deterministic_fallback_embed(text_clean)
        else:
            vec = self._deterministic_fallback_embed(text_clean)

        if self.cache:
            self.cache.put(text_clean, self._model_name, vec)

        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of text strings into a 2D numpy array (N, dimension)."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        # Separate cached from uncached
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        if self.cache:
            for idx, txt in enumerate(texts):
                t_clean = txt.strip()
                cached = self.cache.get(t_clean, self._model_name) if t_clean else np.zeros(self._dimension, dtype=np.float32)
                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(t_clean)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = [t.strip() for t in texts]

        # Compute uncached
        if uncached_texts:
            if self._model is not None:
                try:
                    computed = self._model.encode(
                        uncached_texts,
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    for i, orig_idx in enumerate(uncached_indices):
                        vec = np.array(computed[i], dtype=np.float32)
                        results[orig_idx] = vec
                        if self.cache:
                            self.cache.put(uncached_texts[i], self._model_name, vec)
                except Exception as e:
                    logger.error(f"Batch encoding failed: {e}")
                    for i, orig_idx in enumerate(uncached_indices):
                        vec = self._deterministic_fallback_embed(uncached_texts[i])
                        results[orig_idx] = vec
            else:
                for i, orig_idx in enumerate(uncached_indices):
                    vec = self._deterministic_fallback_embed(uncached_texts[i])
                    results[orig_idx] = vec
                    if self.cache:
                        self.cache.put(uncached_texts[i], self._model_name, vec)

        return np.vstack([r for r in results if r is not None])

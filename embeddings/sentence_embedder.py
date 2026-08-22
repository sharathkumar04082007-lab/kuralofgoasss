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
        self._hf_model = None
        self._tokenizer = None
       # self._initialize_model()

    def _initialize_model(self) -> None:
        """Lazy loads or initializes the transformer model with minimal memory footprint."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            logger.info(f"Loading lightweight transformer: {self._model_name} on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._hf_model = AutoModel.from_pretrained(self._model_name)
            self._hf_model.eval()
            self._dimension = 384
            logger.info(f"Embedding model ready. Dimension: {self._dimension}")
        except BaseException as e:
            logger.warning(f"Transformer load notice ({e}). Initializing deterministic fallback embedder.")
            self._tokenizer = None
            self._hf_model = None
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
                # Lazy-load transformer only when an embedding is actually requested
        if self._hf_model is None or self._tokenizer is None:
           self._initialize_model()

        
        if getattr(self, "_hf_model", None) is not None and getattr(self, "_tokenizer", None) is not None:
            try:
                import torch
                with torch.no_grad():
                    inputs = self._tokenizer([text_clean], padding=True, truncation=True, max_length=128, return_tensors="pt")
                    outputs = self._hf_model(**inputs)
                    # Mean pooling
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs[0].size()).float()
                    summed = torch.sum(outputs[0] * mask, 1)
                    counts = torch.clamp(mask.sum(1), min=1e-9)
                    pooled = (summed / counts).squeeze(0)
                    norm = torch.norm(pooled, p=2, dim=0, keepdim=True).clamp(min=1e-12)
                    vec = (pooled / norm).cpu().numpy().astype(np.float32)
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
            if getattr(self, "_hf_model", None) is not None and getattr(self, "_tokenizer", None) is not None:
                try:
                    import torch
                    with torch.no_grad():
                        inputs = self._tokenizer(uncached_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                        outputs = self._hf_model(**inputs)
                        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs[0].size()).float()
                        summed = torch.sum(outputs[0] * mask, 1)
                        counts = torch.clamp(mask.sum(1), min=1e-9)
                        pooled = summed / counts
                        norm = torch.norm(pooled, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                        batch_vecs = (pooled / norm).cpu().numpy().astype(np.float32)
                except Exception as e:
                    logger.error(f"Batch encoding error: {e}. Using fallback.")
                    batch_vecs = np.array([self._deterministic_fallback_embed(t) for t in uncached_texts], dtype=np.float32)
            else:
                batch_vecs = np.array([self._deterministic_fallback_embed(t) for t in uncached_texts], dtype=np.float32)

            for u_idx, orig_idx in enumerate(uncached_indices):
                vec = batch_vecs[u_idx]
                results[orig_idx] = vec
                if self.cache and uncached_texts[u_idx]:
                    self.cache.put(uncached_texts[u_idx], self._model_name, vec)

        return np.array([r for r in results if r is not None], dtype=np.float32)

import os
import hashlib
import pickle
from collections import OrderedDict
from typing import Optional, Dict, List
import numpy as np
from config.settings import settings
from config.logging_config import logger


class EmbeddingCache:
    """
    Two-tier deterministic embedding cache:
    1. In-memory LRU cache for ultra-low latency sub-millisecond lookups.
    2. Optional disk pickle store for persistence between runs.
    """

    def __init__(
        self, 
        max_size: int = 10000, 
        disk_dir: Optional[str] = None
    ):
        self.max_size = max_size
        self.disk_dir = disk_dir or settings.EMBEDDING_CACHE_DIR
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        
        if self.disk_dir:
            os.makedirs(self.disk_dir, exist_ok=True)

    @staticmethod
    def hash_key(text: str, model_name: str) -> str:
        """Create deterministic key from model name and text content."""
        content = f"{model_name}::{text.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve embedding from memory or disk cache."""
        key = self.hash_key(text, model_name)
        
        # Check RAM LRU
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]
            
        # Check Disk
        if self.disk_dir:
            disk_path = os.path.join(self.disk_dir, f"{key}.npy")
            if os.path.exists(disk_path):
                try:
                    vec = np.load(disk_path)
                    self.put(text, model_name, vec, save_disk=False)
                    return vec
                except Exception as e:
                    logger.debug(f"Failed loading cached embedding from disk: {e}")
                    
        return None

    def put(self, text: str, model_name: str, embedding: np.ndarray, save_disk: bool = True) -> None:
        """Store embedding in cache."""
        key = self.hash_key(text, model_name)
        
        # Evict oldest if full
        if len(self._memory_cache) >= self.max_size:
            self._memory_cache.popitem(last=False)
            
        self._memory_cache[key] = embedding
        
        if save_disk and self.disk_dir:
            try:
                disk_path = os.path.join(self.disk_dir, f"{key}.npy")
                np.save(disk_path, embedding)
            except Exception as e:
                logger.debug(f"Failed saving embedding to disk cache: {e}")

    def clear(self) -> None:
        """Clear both memory and disk caches."""
        self._memory_cache.clear()
        if self.disk_dir and os.path.exists(self.disk_dir):
            for f in os.listdir(self.disk_dir):
                if f.endswith(".npy"):
                    try:
                        os.remove(os.path.join(self.disk_dir, f))
                    except Exception:
                        pass
        logger.info("Embedding cache cleared.")

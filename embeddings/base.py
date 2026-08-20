from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns embedding vector dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate normalized 1D embedding for a single text."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized 2D embeddings for a batch of texts."""
        pass

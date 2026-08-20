from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pipeline.schemas import SourceMetadata


class BaseGenerator(ABC):
    """Abstract interface for LLM answer generation."""

    @abstractmethod
    def generate_answer(self, query: str, sources: List[SourceMetadata]) -> str:
        """Generate answer from query and retrieved source passages."""
        pass

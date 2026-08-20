from typing import Dict, Optional, Any, List
from chunking.base import BaseChunker
from chunking.fixed_chunker import FixedChunker
from chunking.sentence_chunker import SentenceChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.metadata_chunker import MetadataAwareChunker
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import Document, Chunk


class ChunkingStrategySelector:
    """
    Manages and coordinates all chunking strategies.
    Enables runtime strategy selection, fallback mechanisms, and comparative benchmarking.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedder: Optional[Any] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.embedder = embedder

        # Registry of chunkers
        self._chunkers: Dict[str, BaseChunker] = {
            "fixed": FixedChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            "sentence": SentenceChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
            "semantic": SemanticChunker(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap, 
                embedder=self.embedder
            ),
            "metadata": MetadataAwareChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap),
        }

    def get_chunker(self, strategy_name: Optional[str] = None) -> BaseChunker:
        """Get chunker by name or return configured default."""
        strategy = (strategy_name or settings.DEFAULT_CHUNKING_STRATEGY).lower()
        if strategy not in self._chunkers:
            logger.warning(f"Unknown strategy '{strategy}', falling back to 'sentence'")
            strategy = "sentence"
        return self._chunkers[strategy]

    def chunk_documents(
        self, 
        documents: List[Document], 
        strategy_name: Optional[str] = None
    ) -> List[Chunk]:
        """Apply selected chunking strategy on documents with fallback handling."""
        chunker = self.get_chunker(strategy_name)
        try:
            return chunker.chunk_documents(documents)
        except Exception as e:
            logger.error(f"Chunking with strategy '{chunker.strategy_name}' failed: {e}. Falling back to fixed chunker.")
            return self._chunkers["fixed"].chunk_documents(documents)

    def chunk_all_strategies(self, documents: List[Document]) -> Dict[str, List[Chunk]]:
        """Chunk same documents with all 4 strategies for direct A/B comparative evaluation."""
        results: Dict[str, List[Chunk]] = {}
        for name, chunker in self._chunkers.items():
            results[name] = chunker.chunk_documents(documents)
        return results

    @property
    def available_strategies(self) -> List[str]:
        return list(self._chunkers.keys())

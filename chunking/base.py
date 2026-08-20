from abc import ABC, abstractmethod
from typing import List
from pipeline.schemas import Document, Chunk


class BaseChunker(ABC):
    """Abstract Base Class for all chunking strategies."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Returns the identifier string of the chunking strategy."""
        pass

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into multiple Chunks with complete metadata."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Batch chunk a collection of Documents."""
        chunks: List[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

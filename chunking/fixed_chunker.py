from typing import List
from pipeline.schemas import Document, Chunk
from chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    """
    Fixed/Token sliding window chunking strategy.
    Splits text into fixed token/word windows with configurable overlap.
    """

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        position = 0
        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_size]
            chunk_text = " ".join(window)
            
            chunk = Chunk(
                chunk_id=f"{document.document_id}_fixed_{position}",
                document_id=document.document_id,
                parent_document_id=document.document_id,
                text=chunk_text,
                source=document.source,
                language=document.language,
                title=document.title,
                dataset_split=document.split,
                chunking_strategy=self.strategy_name,
                chunk_position=position,
                token_count=len(window),
                character_count=len(chunk_text),
                is_ground_truth=document.metadata.get("is_selected", False),
                metadata={
                    **document.metadata,
                    "window_start": i,
                    "window_end": min(i + self.chunk_size, len(words))
                }
            )
            chunks.append(chunk)
            position += 1
            
            # Avoid creating tiny trailing duplicate chunk
            if i + self.chunk_size >= len(words):
                break

        return chunks

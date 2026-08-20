from typing import List
from pipeline.schemas import Document, Chunk
from chunking.base import BaseChunker


class MetadataAwareChunker(BaseChunker):
    """
    Metadata-aware chunking strategy.
    Preserves document structure by embedding contextual prefixes (such as Query/Title/Passage ID context)
    directly into chunk headers while strictly isolating independent document/passage units from mixing.
    """

    @property
    def strategy_name(self) -> str:
        return "metadata"

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        # Construct structural header from available metadata
        title = document.title or ""
        related_q = document.metadata.get("related_query_en") or document.metadata.get("related_query_indic") or ""
        
        header_parts = []
        if title:
            header_parts.append(f"[Title: {title}]")
        if related_q:
            header_parts.append(f"[Context: {related_q}]")
            
        header = " ".join(header_parts)
        full_context_text = f"{header}\n{text}" if header else text

        words = full_context_text.split()
        if len(words) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{document.document_id}_meta_0",
                    document_id=document.document_id,
                    parent_document_id=document.document_id,
                    text=full_context_text,
                    source=document.source,
                    language=document.language,
                    title=document.title,
                    dataset_split=document.split,
                    chunking_strategy=self.strategy_name,
                    chunk_position=0,
                    token_count=len(words),
                    character_count=len(full_context_text),
                    is_ground_truth=document.metadata.get("is_selected", False),
                    metadata={
                        **document.metadata,
                        "header_injected": bool(header),
                        "has_structural_boundary": True
                    }
                )
            ]

        # For longer documents, split with header retained in every chunk
        chunks: List[Chunk] = []
        body_words = text.split()
        header_words_len = len(header.split()) if header else 0
        effective_chunk_size = max(50, self.chunk_size - header_words_len)
        step = max(1, effective_chunk_size - self.chunk_overlap)
        
        position = 0
        for i in range(0, len(body_words), step):
            window = body_words[i : i + effective_chunk_size]
            body_segment = " ".join(window)
            chunk_content = f"{header}\n{body_segment}" if header else body_segment
            
            chunk = Chunk(
                chunk_id=f"{document.document_id}_meta_{position}",
                document_id=document.document_id,
                parent_document_id=document.document_id,
                text=chunk_content,
                source=document.source,
                language=document.language,
                title=document.title,
                dataset_split=document.split,
                chunking_strategy=self.strategy_name,
                chunk_position=position,
                token_count=len(chunk_content.split()),
                character_count=len(chunk_content),
                is_ground_truth=document.metadata.get("is_selected", False),
                metadata={
                    **document.metadata,
                    "header_injected": bool(header),
                    "chunk_window_start": i,
                    "chunk_window_end": min(i + effective_chunk_size, len(body_words))
                }
            )
            chunks.append(chunk)
            position += 1
            if i + effective_chunk_size >= len(body_words):
                break

        return chunks

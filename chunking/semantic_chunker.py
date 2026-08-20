import re
import numpy as np
from typing import List, Optional
from pipeline.schemas import Document, Chunk
from chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic/Topic-aware chunking strategy.
    Splits text when semantic similarity between consecutive sentence groups drops
    below a threshold, preserving coherent topical discourse.
    """

    SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!।॥\n])\s+')

    def __init__(
        self, 
        chunk_size: int = 250, 
        chunk_overlap: int = 30, 
        similarity_threshold: float = 0.55,
        embedder: Optional[Any] = None
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def _split_sentences(self, text: str) -> List[str]:
        raw = self.SENTENCE_SPLIT_REGEX.split(text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _compute_cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _lexical_jaccard_similarity(self, s1: str, s2: str) -> float:
        """Fast fallback similarity if embedder is not passed or for rapid segmentation."""
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 or not set2:
            return 0.5
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # Single sentence, return as single chunk
            return [
                Chunk(
                    chunk_id=f"{document.document_id}_sem_0",
                    document_id=document.document_id,
                    parent_document_id=document.document_id,
                    text=text,
                    source=document.source,
                    language=document.language,
                    title=document.title,
                    dataset_split=document.split,
                    chunking_strategy=self.strategy_name,
                    chunk_position=0,
                    token_count=len(text.split()),
                    character_count=len(text),
                    is_ground_truth=document.metadata.get("is_selected", False),
                    metadata=document.metadata
                )
            ]

        # Calculate embeddings or similarity shifts between consecutive sentences
        similarities: List[float] = []
        if self.embedder is not None:
            try:
                embeddings = self.embedder.embed_texts(sentences)
                for i in range(len(embeddings) - 1):
                    sim = self._compute_cosine_sim(embeddings[i], embeddings[i + 1])
                    similarities.append(sim)
            except Exception:
                for i in range(len(sentences) - 1):
                    similarities.append(self._lexical_jaccard_similarity(sentences[i], sentences[i + 1]))
        else:
            for i in range(len(sentences) - 1):
                similarities.append(self._lexical_jaccard_similarity(sentences[i], sentences[i + 1]))

        # Group sentences into semantic chunks
        chunks: List[Chunk] = []
        current_sentences: List[str] = [sentences[0]]
        current_word_count = len(sentences[0].split())
        position = 0

        for i in range(len(similarities)):
            next_sentence = sentences[i + 1]
            next_words = len(next_sentence.split())
            sim = similarities[i]

            # Split if:
            # 1. Similarity is below threshold (topic shift) AND current chunk has sufficient words, OR
            # 2. Hard max chunk_size is exceeded
            is_topic_shift = (sim < self.similarity_threshold and current_word_count >= 50)
            is_size_limit = (current_word_count + next_words > self.chunk_size)

            if is_topic_shift or is_size_limit:
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    chunk_id=f"{document.document_id}_sem_{position}",
                    document_id=document.document_id,
                    parent_document_id=document.document_id,
                    text=chunk_text,
                    source=document.source,
                    language=document.language,
                    title=document.title,
                    dataset_split=document.split,
                    chunking_strategy=self.strategy_name,
                    chunk_position=position,
                    token_count=current_word_count,
                    character_count=len(chunk_text),
                    is_ground_truth=document.metadata.get("is_selected", False),
                    metadata={
                        **document.metadata,
                        "semantic_split_score": round(sim, 3),
                        "sentence_count": len(current_sentences)
                    }
                )
                chunks.append(chunk)
                position += 1
                current_sentences = []
                current_word_count = 0

            current_sentences.append(next_sentence)
            current_word_count += next_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                chunk_id=f"{document.document_id}_sem_{position}",
                document_id=document.document_id,
                parent_document_id=document.document_id,
                text=chunk_text,
                source=document.source,
                language=document.language,
                title=document.title,
                dataset_split=document.split,
                chunking_strategy=self.strategy_name,
                chunk_position=position,
                token_count=current_word_count,
                character_count=len(chunk_text),
                is_ground_truth=document.metadata.get("is_selected", False),
                metadata={
                    **document.metadata,
                    "sentence_count": len(current_sentences)
                }
            )
            chunks.append(chunk)

        return chunks

import re
from typing import List
from pipeline.schemas import Document, Chunk
from chunking.base import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Sentence-aware chunking strategy.
    Respects sentence boundaries (Latin . ? ! and Indic । ॥) and aggregates full sentences
    until target chunk size is reached, preserving grammatical integrity.
    """

    # Matches sentences across English and Indic scripts
    SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!।॥\n])\s+')

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving punctuation."""
        raw_sentences = self.SENTENCE_SPLIT_REGEX.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        chunks: List[Chunk] = []
        
        current_sentences: List[str] = []
        current_word_count = 0
        position = 0

        for sent in sentences:
            sent_words = len(sent.split())
            
            # If adding this sentence exceeds chunk_size and we already have content
            if current_sentences and (current_word_count + sent_words > self.chunk_size):
                chunk_text = " ".join(current_sentences)
                chunk = Chunk(
                    chunk_id=f"{document.document_id}_sent_{position}",
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
                position += 1
                
                # Overlap handling: carry over trailing sentences if overlap configured
                if self.chunk_overlap > 0 and len(current_sentences) > 1:
                    overlap_sentences: List[str] = []
                    overlap_words = 0
                    for prev_sent in reversed(current_sentences):
                        prev_w = len(prev_sent.split())
                        if overlap_words + prev_w <= self.chunk_overlap:
                            overlap_sentences.insert(0, prev_sent)
                            overlap_words += prev_w
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_word_count = overlap_words
                else:
                    current_sentences = []
                    current_word_count = 0

            current_sentences.append(sent)
            current_word_count += sent_words

        # Trailing sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = Chunk(
                chunk_id=f"{document.document_id}_sent_{position}",
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

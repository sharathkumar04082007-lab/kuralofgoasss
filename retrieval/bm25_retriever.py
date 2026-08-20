import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from pipeline.schemas import Chunk, SourceMetadata
from config.logging_config import logger


class BM25LexicalRetriever:
    """
    Lexical BM25 retriever for exact keyword, acronym, and entity matching.
    Operates in-memory over indexed chunks.
    """

    TOKEN_REGEX = re.compile(r'\w+', re.UNICODE)

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text handling Indic and Latin scripts."""
        return [t.lower() for t in self.TOKEN_REGEX.findall(text) if len(t) > 1]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build or extend the BM25 inverted index."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokens = [self._tokenize(c.text) for c in chunks]
        self.corpus_tokens.extend(new_tokens)
        
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            logger.info(f"BM25 index built with {len(self.chunks)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy_filter: Optional[str] = None
    ) -> List[SourceMetadata]:
        """Perform BM25 lexical retrieval."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1]

        results: List[SourceMetadata] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
                
            chunk = self.chunks[idx]
            if strategy_filter and chunk.chunking_strategy != strategy_filter:
                continue

            # Normalize BM25 score approximately into [0, 1] range for RRF
            norm_score = float(scores[idx] / (scores[top_indices[0]] + 1e-6))
            
            results.append(
                SourceMetadata(
                    source_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    relevance_score=norm_score,
                    chunking_strategy=chunk.chunking_strategy,
                    text_excerpt=chunk.text,
                    title=chunk.title,
                    language=chunk.language,
                    is_ground_truth=chunk.is_ground_truth
                )
            )
            
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        """Clear BM25 index."""
        self.chunks.clear()
        self.corpus_tokens.clear()
        self.bm25 = None

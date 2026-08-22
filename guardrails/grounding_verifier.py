import re
import numpy as np
from typing import List, Tuple, Optional
from pipeline.schemas import SourceMetadata, GroundingResult
from config.settings import settings
from config.logging_config import logger
from embeddings.sentence_embedder import MultilingualEmbedder


# Standard English and Indic functional stop words to exclude from factual grounding overlap
STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "of", "to", "for", "with",
    "as", "by", "from", "that", "this", "it", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "but", "if", "then", "else", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "what", "who", "whom", "whose", "number", "phone", "telephone", "tell", "give", "please",
    "का", "के", "की", "में", "से", "को", "पर", "है", "हैं", "था", "थे", "थी", "और", "या", "एक", "लिए",
    "क्या", "कैसे", "कब", "कहाँ", "कितना", "कितने", "कितनी", "नंबर", "फोन"
}


class GroundingVerifier:
    """
    Multilingual-first, language-agnostic verification engine that checks whether
    a generated answer is genuinely grounded in and supported by the retrieved context.
    
    Evaluates grounding across four independent signals:
    1. Cross-Lingual Semantic Embedding Cosine Similarity
    2. Symmetric Cross-Lingual Metadata Alignment (Query / Answer linking)
    3. Same-Language Lexical & Bigram Overlap (applied when scripts match)
    4. Dense Retrieval Confidence Score
    """

    REFUSAL_STRING = "I couldn't find enough information in the retrieved data to answer that reliably."

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        threshold: Optional[float] = None,
        semantic_threshold: Optional[float] = None
    ):
        self.embedder = embedder
        self.threshold = threshold if threshold is not None else settings.GROUNDING_THRESHOLD
        self.semantic_threshold = semantic_threshold if semantic_threshold is not None else settings.GROUNDING_SEMANTIC_THRESHOLD

    def _extract_content_tokens(self, text: str) -> List[str]:
        """Extract content tokens, filtering out functional stop words."""
        tokens = [w.lower() for w in re.findall(r'\w+', text, re.UNICODE) if len(w) > 1]
        content_tokens = [t for t in tokens if t not in STOP_WORDS]
        return content_tokens if content_tokens else tokens

    def _extract_bigrams(self, tokens: List[str]) -> set:
        """Extract bigrams from token list."""
        if len(tokens) < 2:
            return set()
        return set(" ".join(tokens[i:i+2]) for i in range(len(tokens)-1))

    def _has_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari unicode characters."""
        return any(0x0900 <= ord(c) <= 0x097F for c in text)

    def _has_latin(self, text: str) -> bool:
        """Check if text contains Latin alphabetic characters."""
        return any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in text)

    def _compute_cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vector representations."""
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def verify_grounding(
        self,
        query: str,
        sources: List[SourceMetadata],
        generated_answer: str
    ) -> GroundingResult:
        """
        Evaluate answer against retrieved sources using language-agnostic cross-lingual verification.
        Returns GroundingResult with status, grounded bool, confidence score, and reasoning.
        """
        if not sources or not generated_answer or generated_answer.strip() == self.REFUSAL_STRING:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=0.0,
                reasoning="No retrieved sources available or model issued explicit refusal."
            )

        answer_clean = generated_answer.strip()
        answer_tokens = self._extract_content_tokens(answer_clean)
        if not answer_tokens:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=0.0,
                reasoning="Generated answer contains no substantive content tokens."
            )

        # 1. Cross-Lingual Metadata Support
        metadata_match_score = 0.0
        q_tokens = set(self._extract_content_tokens(query))
        for src in sources[:5]:
            if src.metadata:
                indic_ans = src.metadata.get("related_answer_indic", "")
                eng_ans = src.metadata.get("related_answer_en", "")
                multi_answers = src.metadata.get("multilingual_answers", {})
                multi_queries = src.metadata.get("multilingual_queries", {})
                
                # Check if answer directly matches any verified multilingual ground-truth answer
                if any(answer_clean == str(v).strip() or str(v).strip() in answer_clean for v in multi_answers.values()):
                    metadata_match_score = 1.0
                    break

                rel_q_tokens = set(self._extract_content_tokens(src.metadata.get("related_query_indic", ""))).union(
                    set(self._extract_content_tokens(src.metadata.get("related_query_en", "")))
                )
                for q_val in multi_queries.values():
                    rel_q_tokens = rel_q_tokens.union(set(self._extract_content_tokens(str(q_val))))

                q_overlap = len(q_tokens.intersection(rel_q_tokens)) / max(1, len(q_tokens)) if q_tokens else 0.0

                all_meta_ans_tokens = set()
                for ans_val in [indic_ans, eng_ans] + list(multi_answers.values()):
                    all_meta_ans_tokens = all_meta_ans_tokens.union(set(self._extract_content_tokens(str(ans_val))))
                
                if all_meta_ans_tokens and (q_overlap >= 0.15 or getattr(src, 'is_ground_truth', False) or src.relevance_score >= 0.50):
                    matched = set(answer_tokens).intersection(all_meta_ans_tokens)
                    cov = len(matched) / len(set(answer_tokens))
                    if cov > metadata_match_score:
                        metadata_match_score = cov

        # If fast metadata match is conclusive (>= 0.50), bypass expensive CPU vector encodings
        if metadata_match_score >= 0.50:
            return GroundingResult(
                status="grounded",
                grounded=True,
                confidence=round(max(0.95, metadata_match_score), 4),
                reasoning=f"High-confidence verified grounded match against indexed ground-truth (coverage: {metadata_match_score:.2f})."
            )

        # 2. Semantic Embedding Similarity (Cross-Lingual)
        max_semantic_sim = 0.0
        query_ans_sim = 1.0
        max_query_context_sim = 0.0
        if self.embedder is not None:
            try:
                ans_vec = self.embedder.embed_text(answer_clean)
                query_vec = self.embedder.embed_text(query)
                query_ans_sim = self._compute_cosine_sim(query_vec, ans_vec)

                for src in sources[:3]:
                    src_text = src.text_excerpt or ""
                    # Combine passage with alternate language answer from metadata if available
                    if src.metadata:
                        alt_ans = src.metadata.get("related_answer_indic") or src.metadata.get("related_answer_en") or ""
                        if alt_ans:
                            src_text = f"{src_text} {alt_ans}"
                    src_vec = self.embedder.embed_text(src_text)
                    sim = self._compute_cosine_sim(ans_vec, src_vec)
                    q_ctx_sim = self._compute_cosine_sim(query_vec, src_vec)
                    
                    if sim > max_semantic_sim:
                        max_semantic_sim = sim
                    if q_ctx_sim > max_query_context_sim:
                        max_query_context_sim = q_ctx_sim
            except Exception as e:
                logger.warning(f"Error computing semantic embedding grounding: {e}")
                max_semantic_sim = 0.5  # Neutral fallback

        # If query and retrieved context are topically disconnected (and no cross-lingual metadata match), refuse
        if self.embedder is not None and max_query_context_sim < 0.55 and metadata_match_score < 0.20:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=round(max_query_context_sim, 3),
                reasoning=f"Retrieved context does not address the query topic (query_context_sim={round(max_query_context_sim, 2)} < 0.55)."
            )

        # If query and generated answer are completely semantically disconnected, refuse
        if self.embedder is not None and query_ans_sim < 0.25 and metadata_match_score < 0.20:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=round(query_ans_sim, 3),
                reasoning=f"Answer is semantically unrelated to query (query_sim={round(query_ans_sim, 2)} < 0.25)."
            )

        # 3. Lexical Token Overlap (Language-Aware)
        combined_context = " ".join([s.text_excerpt for s in sources])
        context_tokens = set(self._extract_content_tokens(combined_context))
        
        answer_is_devanagari = self._has_devanagari(answer_clean)
        context_is_devanagari = self._has_devanagari(combined_context)
        answer_is_latin = self._has_latin(answer_clean)
        context_is_latin = self._has_latin(combined_context)

        same_script = (answer_is_devanagari and context_is_devanagari) or (answer_is_latin and context_is_latin)

        matched_tokens = set(answer_tokens).intersection(context_tokens)
        unigram_overlap = len(matched_tokens) / len(set(answer_tokens))
        
        answer_bigrams = self._extract_bigrams(answer_tokens)
        context_bigrams = self._extract_bigrams(self._extract_content_tokens(combined_context))
        bigram_overlap = len(answer_bigrams.intersection(context_bigrams)) / len(answer_bigrams) if answer_bigrams else unigram_overlap

        # When scripts differ (e.g. Hindi answer generated from English passage), surface unigram overlap is naturally 0.
        # Use cross-lingual metadata coverage and semantic similarity as the primary factual anchors.
        if not same_script:
            lexical_signal = max(metadata_match_score, max_semantic_sim)
        else:
            lexical_signal = 0.60 * unigram_overlap + 0.40 * bigram_overlap

        # 4. Dense Retrieval Relevance
        top_retrieval_score = max([s.relevance_score for s in sources]) if sources else 0.0

        # 5. Composite Multilingual Grounding Score
        if self.embedder is None:
            composite_score = lexical_signal
        else:
            w_sem = settings.GROUNDING_SEMANTIC_WEIGHT
            w_meta = settings.GROUNDING_METADATA_WEIGHT
            w_lex = settings.GROUNDING_LEXICAL_WEIGHT
            w_ret = 0.10
            
            composite_score = (
                w_sem * max_semantic_sim +
                w_meta * max(metadata_match_score, max_semantic_sim * 0.8) +
                w_lex * lexical_signal +
                w_ret * top_retrieval_score
            )
        composite_score = round(min(1.0, max(0.0, composite_score)), 3)

        # Status & Grounding Determination
        if composite_score >= self.threshold and (self.embedder is None or max_semantic_sim >= self.semantic_threshold or metadata_match_score >= 0.30):
            status = "supported"
            grounded = True
            reasoning = f"Answer is strongly supported by retrieved sources (semantic_sim={round(max_semantic_sim, 2)}, composite_score={composite_score})."
        elif composite_score >= (self.threshold * 0.75):
            status = "partially_supported"
            grounded = True
            reasoning = f"Answer is partially supported by retrieved context (composite_score={composite_score})."
        else:
            status = "unsupported"
            grounded = False
            reasoning = f"Answer failed grounding verification: insufficient semantic support (semantic_sim={round(max_semantic_sim, 2)}, score={composite_score} < threshold {self.threshold})."

        return GroundingResult(
            status=status,
            grounded=grounded,
            confidence=composite_score,
            reasoning=reasoning
        )

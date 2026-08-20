import re
from typing import List, Tuple
from pipeline.schemas import SourceMetadata, GroundingResult
from config.settings import settings
from config.logging_config import logger


# Standard English and Indic functional stop words to exclude from factual grounding overlap
STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "of", "to", "for", "with",
    "as", "by", "from", "that", "this", "it", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "but", "if", "then", "else", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "का", "के", "की", "में", "से", "को", "पर", "है", "हैं", "था", "थे", "थी", "और", "या", "एक"
}


class GroundingVerifier:
    """
    Independent verification engine that checks whether a generated answer
    is genuinely grounded in and supported by the retrieved context.
    Filters common stopwords to evaluate substantive factual overlap.
    """

    REFUSAL_STRING = "I couldn't find enough information in the retrieved data to answer that reliably."

    def __init__(self, threshold: float = None):
        self.threshold = threshold if threshold is not None else settings.GROUNDING_THRESHOLD

    def _extract_content_tokens(self, text: str) -> List[str]:
        """Extract content tokens, filtering out stop words."""
        tokens = [w.lower() for w in re.findall(r'\w+', text, re.UNICODE) if len(w) > 1]
        content_tokens = [t for t in tokens if t not in STOP_WORDS]
        return content_tokens if content_tokens else tokens

    def _extract_bigrams(self, tokens: List[str]) -> set:
        """Extract bigrams from token list."""
        if len(tokens) < 2:
            return set()
        return set(" ".join(tokens[i:i+2]) for i in range(len(tokens)-1))

    def verify_grounding(
        self,
        query: str,
        sources: List[SourceMetadata],
        generated_answer: str
    ) -> GroundingResult:
        """
        Evaluate answer against retrieved sources.
        Returns GroundingResult with status, grounded bool, and confidence score.
        """
        if not sources or not generated_answer or generated_answer.strip() == self.REFUSAL_STRING:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=0.0,
                reasoning="No retrieved sources available or model issued explicit refusal."
            )

        combined_context = " ".join([s.text_excerpt for s in sources])
        answer_clean = generated_answer.strip()

        # 1. Content Token Overlap
        answer_tokens = self._extract_content_tokens(answer_clean)
        context_tokens = set(self._extract_content_tokens(combined_context))

        if not answer_tokens:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=0.0,
                reasoning="Generated answer contains no substantive content tokens."
            )

        answer_token_set = set(answer_tokens)
        matched_tokens = answer_token_set.intersection(context_tokens)
        unigram_overlap = len(matched_tokens) / len(answer_token_set)

        # 2. Bigram Overlap
        answer_bigrams = self._extract_bigrams(answer_tokens)
        context_bigrams = self._extract_bigrams(self._extract_content_tokens(combined_context))
        
        if answer_bigrams:
            bigram_overlap = len(answer_bigrams.intersection(context_bigrams)) / len(answer_bigrams)
        else:
            bigram_overlap = unigram_overlap

        # If substantive unigram overlap is below 40%, it is not adequately grounded
        if unigram_overlap < 0.40:
            return GroundingResult(
                status="unsupported",
                grounded=False,
                confidence=round(unigram_overlap, 3),
                reasoning=f"Answer contains key factual assertions not present in retrieved context (overlap: {round(unigram_overlap * 100, 1)}%)."
            )

        # 3. Composite Grounding Score
        top_retrieval_score = max([s.relevance_score for s in sources]) if sources else 0.0
        composite_score = (
            0.55 * unigram_overlap +
            0.30 * bigram_overlap +
            0.15 * min(1.0, top_retrieval_score)
        )
        composite_score = round(min(1.0, max(0.0, composite_score)), 3)

        # Status determination
        if composite_score >= self.threshold:
            status = "supported"
            grounded = True
            reasoning = f"Answer is strongly supported by retrieved sources (overlap={round(unigram_overlap, 2)}, score={composite_score})."
        elif composite_score >= (self.threshold * 0.75):
            status = "partially_supported"
            grounded = True
            reasoning = f"Answer is partially supported by retrieved context (score={composite_score})."
        else:
            status = "unsupported"
            grounded = False
            reasoning = f"Answer failed grounding verification: hallucination risk (score={composite_score} < threshold {self.threshold})."

        return GroundingResult(
            status=status,
            grounded=grounded,
            confidence=composite_score,
            reasoning=reasoning
        )

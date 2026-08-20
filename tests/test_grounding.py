import pytest
from pipeline.schemas import SourceMetadata
from guardrails.grounding_verifier import GroundingVerifier


@pytest.fixture
def sample_sources():
    return [
        SourceMetadata(
            source_id="src_1",
            document_id="doc_1",
            relevance_score=0.92,
            chunking_strategy="sentence",
            text_excerpt="Paris is the capital and most populous city of France, situated along the Seine River.",
            title="Capital of France"
        )
    ]


def test_grounded_answer(sample_sources):
    verifier = GroundingVerifier(threshold=0.35)
    query = "what is the capital of France?"
    answer = "Paris is the capital of France, located along the Seine River."
    result = verifier.verify_grounding(query, sample_sources, answer)
    
    assert result.grounded is True
    assert result.status in ["supported", "partially_supported"]
    assert result.confidence >= 0.35


def test_hallucinated_unsupported_answer(sample_sources):
    verifier = GroundingVerifier(threshold=0.35)
    query = "what is the capital of France?"
    # Hallucination unrelated to context
    answer = "The capital of France is Atlantis, famous for its underwater dolphin palaces."
    result = verifier.verify_grounding(query, sample_sources, answer)
    
    assert result.grounded is False
    assert result.status == "unsupported"


def test_refusal_when_no_sources():
    verifier = GroundingVerifier()
    result = verifier.verify_grounding("query", [], "Some answer")
    assert result.grounded is False
    assert result.confidence == 0.0

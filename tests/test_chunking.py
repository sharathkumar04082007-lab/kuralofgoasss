import pytest
from pipeline.schemas import Document
from chunking.fixed_chunker import FixedChunker
from chunking.sentence_chunker import SentenceChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.metadata_chunker import MetadataAwareChunker
from chunking.selector import ChunkingStrategySelector


@pytest.fixture
def sample_document():
    return Document(
        document_id="doc_test_101",
        title="Test Document",
        text="Photosynthesis is the process by which green plants convert light into chemical energy. "
             "This occurs primarily inside the chloroplasts of plant cells. "
             "Chlorophyll absorbs solar radiation to drive the chemical reaction. "
             "Oxygen is released as a vital byproduct into the atmosphere.",
        language="en",
        split="validation",
        metadata={"is_selected": True, "related_query_en": "what is photosynthesis?"}
    )


def test_fixed_chunker(sample_document):
    chunker = FixedChunker(chunk_size=15, chunk_overlap=3)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    assert chunks[0].chunking_strategy == "fixed"
    assert chunks[0].document_id == "doc_test_101"
    assert chunks[0].is_ground_truth is True


def test_sentence_chunker(sample_document):
    chunker = SentenceChunker(chunk_size=25, chunk_overlap=5)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert chunks[0].chunking_strategy == "sentence"
    # Ensure sentence boundary was preserved
    assert chunks[0].text.endswith(".")


def test_semantic_chunker(sample_document):
    chunker = SemanticChunker(chunk_size=20, similarity_threshold=0.6)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert chunks[0].chunking_strategy == "semantic"


def test_metadata_chunker(sample_document):
    chunker = MetadataAwareChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert chunks[0].chunking_strategy == "metadata"
    # Check that metadata header context is injected
    assert "[Title: Test Document]" in chunks[0].text
    assert "[Context: what is photosynthesis?]" in chunks[0].text


def test_selector_all_strategies(sample_document):
    selector = ChunkingStrategySelector()
    results = selector.chunk_all_strategies([sample_document])
    assert "fixed" in results
    assert "sentence" in results
    assert "semantic" in results
    assert "metadata" in results
    for strat, chunks in results.items():
        assert len(chunks) > 0


def test_empty_document_handling():
    empty_doc = Document(document_id="empty", text="", language="en")
    selector = ChunkingStrategySelector()
    chunks = selector.chunk_documents([empty_doc], strategy_name="sentence")
    assert len(chunks) == 0

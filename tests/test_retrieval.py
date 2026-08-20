import pytest
import numpy as np
from pipeline.schemas import Chunk
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever
from retrieval.hybrid_retriever import HybridRetriever
from embeddings.sentence_embedder import MultilingualEmbedder


@pytest.fixture(scope="module")
def embedder():
    return MultilingualEmbedder()


@pytest.fixture(scope="module")
def vector_store(embedder):
    store = QdrantVectorStore(
        collection_name="test_retrieval_col", 
        dimension=embedder.dimension,
        storage_path=":memory:"
    )
    store.clear()
    return store


@pytest.fixture(scope="module")
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk_paris_1",
            document_id="doc_paris",
            parent_document_id="doc_paris",
            text="Paris is the capital and largest city of France with historic landmarks like the Eiffel Tower.",
            language="en",
            chunking_strategy="sentence",
            is_ground_truth=True
        ),
        Chunk(
            chunk_id="chunk_tokyo_1",
            document_id="doc_tokyo",
            parent_document_id="doc_tokyo",
            text="Tokyo is the bustling capital of Japan, famous for its advanced technology and cherry blossoms.",
            language="en",
            chunking_strategy="sentence",
            is_ground_truth=False
        ),
        Chunk(
            chunk_id="chunk_mars_1",
            document_id="doc_mars",
            parent_document_id="doc_mars",
            text="Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System.",
            language="en",
            chunking_strategy="sentence",
            is_ground_truth=False
        ),
    ]


def test_qdrant_vector_store(vector_store, embedder, sample_chunks):
    texts = [c.text for c in sample_chunks]
    embs = embedder.embed_texts(texts)
    upsert_ok = vector_store.upsert_chunks(sample_chunks, embs)
    assert upsert_ok is True
    assert vector_store.count() >= 3

    # Dense query search
    q_vec = embedder.embed_text("what is the capital of France?")
    results = vector_store.search(q_vec, top_k=2)
    assert len(results) >= 1
    assert "Paris" in results[0].text_excerpt


def test_bm25_retriever(sample_chunks):
    bm25 = BM25LexicalRetriever()
    bm25.index_chunks(sample_chunks)
    
    hits = bm25.search("capital Japan Tokyo", top_k=1)
    assert len(hits) == 1
    assert hits[0].document_id == "doc_tokyo"


def test_hybrid_retriever(vector_store, embedder, sample_chunks):
    bm25 = BM25LexicalRetriever()
    bm25.index_chunks(sample_chunks)
    
    hybrid = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25, embedder=embedder)
    results = hybrid.retrieve("capital of France", top_k=2)
    assert len(results) >= 1
    assert results[0].document_id == "doc_paris"

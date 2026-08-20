from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import time


class Document(BaseModel):
    """Normalized document representation from MSMARCO-XI."""
    document_id: str = Field(..., description="Unique deterministic document ID")
    query_id: Optional[int] = Field(None, description="Original query ID if from Q&A dataset")
    title: Optional[str] = Field(default="", description="Document or passage title")
    text: str = Field(..., description="Cleaned normalized document text")
    language: str = Field(default="en", description="ISO language code (e.g. en, hi, ta, etc.)")
    source: str = Field(default="ai4bharat/MSMARCO-XI", description="Dataset source")
    split: str = Field(default="train", description="Dataset split (train, validation)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class Chunk(BaseModel):
    """Normalized chunk model storing complete metadata."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document identifier")
    parent_document_id: str = Field(..., description="Explicit parent reference")
    text: str = Field(..., description="Chunk content text")
    source: str = Field(default="ai4bharat/MSMARCO-XI", description="Source dataset")
    language: str = Field(default="en", description="Language code")
    title: Optional[str] = Field(default="", description="Document title if available")
    dataset_split: str = Field(default="train", description="Split name")
    chunking_strategy: str = Field(..., description="fixed, sentence, semantic, or metadata")
    chunk_position: int = Field(default=0, description="0-indexed position within document")
    token_count: int = Field(default=0, description="Approximated or exact token count")
    character_count: int = Field(default=0, description="Length of chunk in characters")
    is_ground_truth: bool = Field(default=False, description="Whether marked as relevant passage in dataset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary metadata")


class SourceMetadata(BaseModel):
    """Retrieved source chunk representation displayed to user and frontend."""
    source_id: str
    document_id: str
    relevance_score: float
    chunking_strategy: str
    text_excerpt: str
    title: Optional[str] = ""
    language: str = "en"
    is_ground_truth: Optional[bool] = None


class LatencyBreakdown(BaseModel):
    """Fine-grained latency tracking by pipeline stage (in milliseconds)."""
    stt_ms: float = 0.0
    guardrails_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_ms: float = 0.0


class GroundingResult(BaseModel):
    """Verification result for hallucination and factual grounding."""
    status: str = Field(..., description="supported, partially_supported, or unsupported")
    grounded: bool = Field(..., description="True if answer is faithfully supported by retrieved context")
    confidence: float = Field(..., description="Grounding confidence score between 0.0 and 1.0")
    reasoning: Optional[str] = Field(default="", description="Explanation of grounding decision")


class QueryRequest(BaseModel):
    """Text query request payload."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    chunking_strategy: Optional[str] = Field(default=None, description="Optional chunking strategy filter")
    use_reranker: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(default="en")


class QueryResponse(BaseModel):
    """Complete structured response conforming to project specification."""
    transcript: str = Field(..., description="Transcribed audio or input query")
    answer: str = Field(..., description="Grounded synthesized answer")
    sources: List[SourceMetadata] = Field(default_factory=list, description="Retrieved source passages")
    confidence: float = Field(..., description="Calculated overall confidence score [0.0 - 1.0]")
    grounded: bool = Field(..., description="Whether answer is validated against retrieved context")
    latency_ms: LatencyBreakdown = Field(..., description="Latency breakdown across all pipeline stages")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request tracing ID")
    query_classification: str = Field(default="valid", description="valid, off_topic, unsafe, or ambiguous")
    error: Optional[str] = Field(default=None, description="Error message if pipeline encountered non-fatal issue")


class IngestionProgress(BaseModel):
    """Progress tracker for dataset ingestion."""
    records_processed: int = 0
    chunks_indexed: int = 0
    languages: Dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    status: str = "idle"
    last_checkpoint: Optional[str] = None

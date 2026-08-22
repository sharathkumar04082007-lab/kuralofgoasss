import os
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Environment mode
    ENVIRONMENT: str = Field(default="development", description="development or production")
    APP_NAME: str = Field(default="VoiceRAG-MSMARCO-XI", description="Application Name")
    APP_VERSION: str = Field(default="1.0.0", description="Application Version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server Host")
    PORT: int = Field(default=8000, description="Server Port")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")
    
    # Speech-to-Text Provider (sarvam, elevenlabs, or mock)
    STT_PROVIDER: str = Field(default="sarvam", description="sarvam, elevenlabs, or mock")
    SARVAM_API_KEY: Optional[str] = Field(default=None, description="Sarvam AI API subscription key")
    ELEVENLABS_API_KEY: Optional[str] = Field(default=None, description="ElevenLabs API key")
    STT_LANGUAGE_CODE: str = Field(default="unknown", description="Language code for ASR: unknown for auto-detect, hi-IN, en-IN")
    STT_TIMEOUT_SECONDS: float = Field(default=8.0, description="HTTP timeout for STT requests")

    # Text-to-Speech Provider (sarvam or mock)
    TTS_PROVIDER: str = Field(default="sarvam", description="sarvam or mock")
    TTS_SPEAKER: str = Field(default="meera", description="Sarvam TTS voice speaker: meera, arvind, amartya, ananya")
    TTS_LANGUAGE_CODE: str = Field(default="hi-IN", description="Default language code for TTS: hi-IN, en-IN")
    
    # LLM / Answer Generation configuration
    LLM_PROVIDER: str = Field(default="groq", description="local, groq, gemini, or mock")
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API Key for ultra-fast inference")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API Key")
    LLM_MODEL: str = Field(default="groq/compound-mini", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.1, description="Generation temperature")
    LLM_MAX_TOKENS: int = Field(default=150, description="Max generated tokens for low latency")
    LLM_TIMEOUT_SECONDS: float = Field(default=3.0, description="Generation timeout in seconds")
    
    # Embedding configuration
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Multilingual sentence transformer for Indic + English"
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", description="cpu or cuda")
    EMBEDDING_CACHE_SIZE: int = Field(default=10000, description="LRU RAM cache capacity for embeddings")
    EMBEDDING_CACHE_DIR: str = Field(default="./data/cache/embeddings", description="Disk cache dir for embeddings")
    
    # Vector Database (Qdrant)
    QDRANT_MODE: str = Field(default="embedded", description="embedded or remote")
    QDRANT_PATH: str = Field(default="./data/qdrant_db", description="Local path for embedded Qdrant storage")
    QDRANT_URL: Optional[str] = Field(default=None, description="Remote Qdrant URL (e.g. http://localhost:6333)")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API Key if remote")
    QDRANT_COLLECTION_NAME: str = Field(default="msmarco_xi_collection", description="Default Qdrant collection name")
    
    # Retrieval parameters
    TOP_K: int = Field(default=5, description="Number of context chunks to retrieve")
    RERANK_TOP_K: int = Field(default=15, description="Initial candidates to retrieve before reranking")
    MIN_RETRIEVAL_SCORE: float = Field(default=0.30, description="Minimum cosine similarity threshold")
    HYBRID_SEARCH: bool = Field(default=True, description="Enable hybrid dense + BM25 retrieval")
    USE_RERANKER: bool = Field(default=False, description="Enable cross-encoder reranker (false for sub-200ms path)")
    
    # Chunking strategy
    DEFAULT_CHUNKING_STRATEGY: str = Field(default="sentence", description="fixed, sentence, semantic, or metadata")
    CHUNK_SIZE: int = Field(default=250, description="Target token / word length per chunk")
    CHUNK_OVERLAP: int = Field(default=30, description="Overlap token count")
    
    # Guardrails and Grounding
    ENABLE_SAFETY_GUARDRAILS: bool = Field(default=True, description="Enable safety and toxic content filter")
    ENABLE_PROMPT_INJECTION_DEFENSE: bool = Field(default=True, description="Sanitize query and context injections")
    GROUNDING_THRESHOLD: float = Field(default=0.35, description="Minimum grounding entailment score required to answer")
    GROUNDING_SEMANTIC_THRESHOLD: float = Field(default=0.55, description="Minimum semantic cross-lingual embedding similarity threshold")
    PRE_GEN_DENSE_MIN_SCORE: float = Field(default=0.55, description="Minimum dense score for pre-generation sufficiency")
    GROUNDING_SEMANTIC_WEIGHT: float = Field(default=0.50, description="Weight for semantic embedding similarity")
    GROUNDING_METADATA_WEIGHT: float = Field(default=0.30, description="Weight for cross-lingual metadata support")
    GROUNDING_LEXICAL_WEIGHT: float = Field(default=0.20, description="Weight for same-language lexical overlap")
    MAX_QUERY_LENGTH: int = Field(default=300, description="Maximum characters allowed in query")
    
    # Dataset Ingestion
    DATASET_NAME: str = Field(default="ai4bharat/MSMARCO-XI", description="Dataset identifier")
    DATASET_CACHE_DIR: str = Field(default="./data/msmarco_xi", description="Local dataset cache folder")
    
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

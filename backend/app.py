import os
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import QueryRequest, QueryResponse
from pipeline.orchestrator import RAGOrchestrator
from analytics.metrics_collector import metrics_collector
from chunking.selector import ChunkingStrategySelector

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production Voice-Enabled RAG System on MSMARCO-XI with Sub-200ms Target Architecture"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Orchestrator instance
orchestrator: Optional[RAGOrchestrator] = None


@app.on_event("startup")
async def startup_event():
    global orchestrator
    logger.info("Starting up FastAPI Voice RAG Server...")
    orchestrator = RAGOrchestrator()
    logger.info("FastAPI Voice RAG Server initialized.")


@app.get("/api/health")
async def health():
    """Health check and vector store status."""
    global orchestrator
    if orchestrator is None:
        return {"status": "initializing"}
    
    db_health = orchestrator.vector_store.health_check()
    bm25_count = len(orchestrator.bm25_retriever.chunks)
    
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "vector_store": db_health,
        "bm25_indexed_chunks": bm25_count,
        "stt_provider": orchestrator.stt_provider.provider_name,
        "llm_provider": orchestrator.generator.provider
    }


@app.get("/api/config")
async def get_config():
    """Get active pipeline configuration and chunking strategy options."""
    selector = ChunkingStrategySelector()
    return {
        "active_chunking_strategy": settings.DEFAULT_CHUNKING_STRATEGY,
        "available_strategies": selector.available_strategies,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "vector_db_mode": settings.QDRANT_MODE,
        "hybrid_search": settings.HYBRID_SEARCH,
        "use_reranker": settings.USE_RERANKER,
        "stt_provider": settings.STT_PROVIDER,
        "llm_provider": settings.LLM_PROVIDER,
        "min_retrieval_score": settings.MIN_RETRIEVAL_SCORE,
        "top_k": settings.TOP_K
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get latency analytics and P50/P70/P100 percentiles."""
    summary = metrics_collector.generate_summary()
    return summary


@app.post("/api/text/query", response_model=QueryResponse)
async def text_query(request: QueryRequest):
    """
    Process text query through RAG pipeline.
    """
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")

    try:
        response = orchestrator.process_text_query(request)
        return response
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(..., description="Audio file in WAV, WebM, or PCM format"),
    language_code: Optional[str] = Form(default="hi-IN"),
    top_k: Optional[int] = Form(default=5),
    chunking_strategy: Optional[str] = Form(default=None),
    use_reranker: Optional[bool] = Form(default=False)
):
    """
    Process voice audio input through end-to-end Voice RAG pipeline:
    Voice Input -> STT -> Hybrid Retrieval -> Answer Generation -> Grounding Check.
    """
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes or len(audio_bytes) < 10:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")

        response = orchestrator.process_voice_query(
            audio_bytes=audio_bytes,
            language_code=language_code,
            filename=audio.filename or "audio.wav",
            top_k=top_k,
            chunking_strategy=chunking_strategy,
            use_reranker=use_reranker
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Serve static Frontend files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

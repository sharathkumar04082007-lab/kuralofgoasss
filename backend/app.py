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
from qdrant_client.http import models as rest_models

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
    if orchestrator is None:
        logger.info("Starting up FastAPI Voice RAG Server...")
        try:
            orchestrator = RAGOrchestrator()
        except RuntimeError as e:
            if "already accessed" in str(e):
                from retrieval.vector_store import QdrantVectorStore
                mem_store = QdrantVectorStore(storage_path=":memory:")
                orchestrator = RAGOrchestrator(vector_store=mem_store)
            else:
                raise

        # Auto-index seed knowledge base and 5-language multilingual demo suite
        from scripts.index_multilingual_demo_suite import DEMO_MULTILINGUAL_RECORDS
        from pipeline.schemas import Chunk

        demo_chunks = []
        demo_texts = []
        for item in DEMO_MULTILINGUAL_RECORDS:
            qid = item["query_id"]
            title = item["title"]
            queries = item["queries"]
            answers = item["answers"]
            passages_by_lang = item["passages"]

            for lang, pass_list in passages_by_lang.items():
                for p_idx, p_text in enumerate(pass_list):
                    c_id = f"demo_{qid}_{lang}_{p_idx}"
                    chunk = Chunk(
                        chunk_id=c_id,
                        document_id=f"doc_{qid}_{lang}",
                        parent_document_id=f"doc_{qid}",
                        text=p_text,
                        source="ai4bharat/MSMARCO-XI",
                        language=lang,
                        title=f"{title} ({lang})",
                        dataset_split="validation",
                        chunking_strategy="sentence",
                        chunk_position=p_idx,
                        token_count=len(p_text.split()),
                        character_count=len(p_text),
                        is_ground_truth=(p_idx == 0),
                        metadata={
                            "query_id": qid,
                            "passage_index": p_idx,
                            "is_selected": (p_idx == 0),
                            "related_query_en": queries.get("en", ""),
                            "related_query_indic": queries.get(lang, queries.get("hi", "")),
                            "related_answer_en": answers.get("en", ""),
                            "related_answer_indic": answers.get(lang, answers.get("hi", "")),
                            "multilingual_queries": queries,
                            "multilingual_answers": answers
                        }
                    )
                    demo_chunks.append(chunk)
                    demo_texts.append(p_text)

        logger.info(f"Embedding and indexing {len(demo_chunks)} multilingual demo chunks (en, hi, kn, ta, te)...")
        demo_embeddings = orchestrator.embedder.embed_texts(demo_texts)
        orchestrator.vector_store.upsert_chunks(demo_chunks, demo_embeddings)
        orchestrator.bm25_retriever.index_chunks(demo_chunks)

        if len(orchestrator.bm25_retriever.chunks) == len(demo_chunks):
            logger.info("Auto-indexing MSMARCO-XI dataset background records...")
            from ingestion.pipeline import IngestionPipeline
            from ingestion.normalizer import TextNormalizer
            from ingestion.dataset_loader import MSMARCODatasetLoader, OFFLINE_SAMPLE_RECORDS
            
            loader = MSMARCODatasetLoader()
            docs = []
            for rec in OFFLINE_SAMPLE_RECORDS:
                docs.extend(TextNormalizer.parse_msmarco_record(rec))
            try:
                for rec in loader.stream_records(languages=["hi"], max_records=80, use_fallback_if_offline=False):
                    docs.extend(TextNormalizer.parse_msmarco_record(rec))
            except Exception as e:
                logger.warning(f"Notice streaming MSMARCO parquet: {e}")
            
            ingestor = IngestionPipeline(
                embedder=orchestrator.embedder,
                vector_store=orchestrator.vector_store,
                bm25_retriever=orchestrator.bm25_retriever
            )
            ingestor._process_and_index_batch(
                documents=docs,
                strategy=settings.DEFAULT_CHUNKING_STRATEGY,
                progress=type("obj", (object,), {"records_processed": 0, "chunks_indexed": 0})()
            )
            logger.info(f"Auto-indexed {len(orchestrator.bm25_retriever.chunks)} chunks into vector store & BM25 index.")

        # Pre-warm demo audio synthesis and resident memory cache for Sub-200ms P100 execution
        logger.info("Pre-warming demo multilingual TTS audio cache (EN, HI, KN, TA, TE)...")
        for item in DEMO_MULTILINGUAL_RECORDS:
            answers = item.get("answers", {})
            for lang, ans_text in answers.items():
                bcp = f"{lang}-IN" if lang != "en" else "en-IN"
                try:
                    orchestrator.tts_provider.synthesize(ans_text, language_code=bcp)
                except Exception as e:
                    logger.warning(f"Notice pre-warming TTS for [{lang}]: {e}")

        logger.info("FastAPI Voice RAG Server initialized with 5-language resident knowledge base and pre-warmed audio cache.")


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
    language_code: Optional[str] = Form(default="unknown"),
    browser_transcript: Optional[str] = Form(default=None),
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
        logger.info(
            f"Received audio:\n"
            f"filename = {audio.filename}\n"
            f"content_type = {audio.content_type}\n"
            f"size = {len(audio_bytes)} bytes\n"
            f"browser_transcript = {repr(browser_transcript)}"
        )

        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "debug_audio")
        os.makedirs(debug_dir, exist_ok=True)
        ext = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
        debug_file_path = os.path.join(debug_dir, f"last_received{ext}")
        with open(debug_file_path, "wb") as f:
            f.write(audio_bytes)

        if not audio_bytes or len(audio_bytes) < 10:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")

        response = orchestrator.process_voice_query(
            audio_bytes=audio_bytes,
            language_code=language_code,
            filename=audio.filename or "audio.webm",
            browser_transcript=browser_transcript,
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


# Diagnostic & RAG Deep Inspection Endpoints
@app.get("/api/diagnostics/inspect_payload")
async def inspect_payload(query_id: Optional[int] = None, limit: int = 10):
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")

    client = orchestrator.vector_store.client
    collection_name = orchestrator.vector_store.collection_name
    
    # Scroll points
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=2000,
        with_payload=True,
        with_vectors=False
    )
    
    total_in_db = len(points)
    matching_points = []
    
    for p in points:
        payload = p.payload or {}
        doc_qid = payload.get("metadata", {}).get("query_id") or payload.get("query_id")
        doc_id = payload.get("document_id", "")
        
        # Check if query_id is in doc_id or metadata
        if query_id is not None:
            if doc_qid == query_id or f"_{query_id}_" in doc_id:
                matching_points.append({
                    "point_id": p.id,
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": doc_id,
                    "query_id": doc_qid or (query_id if f"_{query_id}_" in doc_id else None),
                    "title": payload.get("title"),
                    "language": payload.get("language"),
                    "text": payload.get("text"),
                    "metadata": payload.get("metadata"),
                    "is_ground_truth": payload.get("is_ground_truth")
                })
        else:
            if len(matching_points) < limit:
                matching_points.append({
                    "point_id": p.id,
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": doc_id,
                    "query_id": doc_qid,
                    "title": payload.get("title"),
                    "language": payload.get("language"),
                    "text": payload.get("text"),
                    "metadata": payload.get("metadata"),
                    "is_ground_truth": payload.get("is_ground_truth")
                })

    return {
        "total_points_in_db": total_in_db,
        "query_id_searched": query_id,
        "matched_count": len(matching_points),
        "sample_points": matching_points[:limit]
    }


@app.post("/api/diagnostics/run_test")
async def run_diagnostic_test(data: dict):
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")

    query = data.get("query", "")
    expected_query_id = data.get("expected_query_id")
    top_k = data.get("top_k", 10)
    use_reranker = data.get("use_reranker", False)
    retrieval_mode = data.get("retrieval_mode", "dense")  # dense, bm25, hybrid, or pipeline

    # 1. Original and Preprocessed Query
    classification, explanation = orchestrator.classifier.classify(query)
    
    # 2. Embedding Model & Vector DB Collection
    embedding_model = settings.EMBEDDING_MODEL_NAME
    collection_name = orchestrator.vector_store.collection_name

    # 3. Retrieval
    query_vector = orchestrator.embedder.embed_text(query)
    retrieved_sources = []
    
    if retrieval_mode == "dense":
        retrieved_sources = orchestrator.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=0.0  # Show all top_k without cutoff for deep diagnosis
        )
    elif retrieval_mode == "bm25":
        retrieved_sources = orchestrator.bm25_retriever.search(
            query=query,
            top_k=top_k
        )
    elif retrieval_mode == "hybrid":
        # Hybrid dense + bm25
        dense_c = orchestrator.vector_store.search(query_vector=query_vector, top_k=top_k * 2, min_score=0.0)
        bm25_c = orchestrator.bm25_retriever.search(query=query, top_k=top_k * 2)
        
        rrf_scores = {}
        sources_by_id = {}
        for rank, item in enumerate(dense_c):
            sources_by_id[item.source_id] = item
            rrf_scores[item.source_id] = rrf_scores.get(item.source_id, 0.0) + 0.7 / (60 + rank + 1)
        for rank, item in enumerate(bm25_c):
            if item.source_id not in sources_by_id:
                sources_by_id[item.source_id] = item
            rrf_scores[item.source_id] = rrf_scores.get(item.source_id, 0.0) + 0.3 / (60 + rank + 1)
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        for sid in sorted_ids[:top_k]:
            src = sources_by_id[sid]
            src.relevance_score = round(rrf_scores[sid] * 61, 4)
            retrieved_sources.append(src)
    else:
        # Full pipeline retrieval
        retrieved_sources = orchestrator.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            min_score=settings.MIN_RETRIEVAL_SCORE
        )

    # 4. Optional Reranker
    reranked_sources = retrieved_sources
    if use_reranker and len(retrieved_sources) > 0:
        reranked_sources = orchestrator.reranker.rerank(
            query=query,
            candidates=retrieved_sources,
            top_k=top_k
        )

    active_sources = reranked_sources if use_reranker else retrieved_sources

    # 5. Extract results details
    detailed_results = []
    correct_retrieved = False
    correct_rank = None

    for rank, src in enumerate(active_sources, 1):
        # Extract query_id from doc_id or stored metadata
        extracted_qid = None
        if src.document_id:
            import re as rex
            m = rex.search(r'doc_(\d+)_', src.document_id)
            if m:
                extracted_qid = int(m.group(1))

        # Check if matched expected query_id
        is_target = False
        if expected_query_id is not None and extracted_qid == expected_query_id:
            is_target = True
            if not correct_retrieved:
                correct_retrieved = True
                correct_rank = rank

        # Fetch point payload from Qdrant to get stored question/answer if available
        stored_q = ""
        stored_a = ""
        try:
            # Query point by source_id/chunk_id
            pts = orchestrator.vector_store.client.scroll(
                collection_name=collection_name,
                scroll_filter=rest_models.Filter(
                    must=[rest_models.FieldCondition(key="chunk_id", match=rest_models.MatchValue(value=src.source_id))]
                ),
                limit=1,
                with_payload=True
            )[0]
            if pts:
                p_meta = pts[0].payload.get("metadata", {})
                stored_q = p_meta.get("related_query_indic") or p_meta.get("related_query_en") or ""
                stored_a = p_meta.get("related_answer_indic") or p_meta.get("related_answer_en") or ""
        except Exception:
            pass

        detailed_results.append({
            "rank": rank,
            "query_id": extracted_qid,
            "document_id": src.document_id,
            "chunk_id": src.source_id,
            "similarity_score": round(src.relevance_score, 4),
            "stored_query": stored_q,
            "stored_answer": stored_a,
            "relevant_passage": src.text_excerpt,
            "is_target_query_id": is_target,
            "is_ground_truth": src.is_ground_truth
        })

    # 6. Exact context passed to LLM
    context_passed = "\n\n".join([f"[{s.title or s.document_id}]: {s.text_excerpt}" for s in active_sources[:settings.TOP_K]])
    prompt_passed = orchestrator.generator.prompt_builder.build_prompt(query, active_sources[:settings.TOP_K])

    # 7. Generation
    generated_answer = orchestrator.generator.generate_answer(query=query, sources=active_sources[:settings.TOP_K])

    # 8. Grounding Verification
    grounding_res = orchestrator.grounding_verifier.verify_grounding(
        query=query,
        sources=active_sources[:settings.TOP_K],
        generated_answer=generated_answer
    )

    # 9. Final application answer
    final_answer = generated_answer
    if not grounding_res.grounded:
        final_answer = orchestrator.REFUSAL_MESSAGE

    return {
        "original_query": query,
        "preprocessed_query": query.strip(),
        "query_classification": classification,
        "embedding_model": embedding_model,
        "vector_db_collection": collection_name,
        "retrieval_mode": retrieval_mode,
        "use_reranker": use_reranker,
        "top_10_results": detailed_results,
        "correct_query_id_retrieved": correct_retrieved,
        "rank_of_correct_query_id": correct_rank,
        "exact_context_passed_to_llm": context_passed,
        "exact_prompt_passed_to_llm": prompt_passed,
        "generated_answer": generated_answer,
        "grounding_result": {
            "status": grounding_res.status,
            "grounded": grounding_res.grounded,
            "confidence": grounding_res.confidence,
            "reasoning": grounding_res.reasoning
        },
        "final_application_answer": final_answer
    }


# Serve static Frontend files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


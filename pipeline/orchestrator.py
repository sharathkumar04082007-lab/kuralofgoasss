import time
import uuid
import re
from typing import Optional, List, Dict, Any
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import (
    QueryRequest, 
    QueryResponse, 
    SourceMetadata, 
    LatencyBreakdown,
    GroundingResult
)
from analytics.latency_tracker import LatencyTracker
from analytics.metrics_collector import metrics_collector
from guardrails.safety_filter import SafetyGuardrails
from guardrails.query_classifier import QueryClassifier
from guardrails.grounding_verifier import GroundingVerifier, STOP_WORDS
from embeddings.sentence_embedder import MultilingualEmbedder
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.llm_generator import LLMAnswerGenerator
from stt.base import SpeechToTextProvider
from stt.factory import STTProviderFactory


class RAGOrchestrator:
    """
    Main Production Harness & Orchestrator.
    Controls the entire voice and text RAG pipeline:
    STT -> Safety -> Query Classification -> Hybrid Retrieval -> Reranking -> 
    Answer Generation -> Grounding Verification -> Response Packaging.
    """

    REFUSAL_MESSAGE = "I couldn't find enough information in the retrieved data to answer that reliably."
    UNSAFE_MESSAGE = "I cannot fulfill this request as it violates safety guidelines."
    OFF_TOPIC_MESSAGE = "I am a specialized MSMARCO factual assistant. Please ask a factual question related to the indexed knowledge base."

    def __init__(
        self,
        embedder: Optional[MultilingualEmbedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_retriever: Optional[BM25LexicalRetriever] = None,
        stt_provider: Optional[SpeechToTextProvider] = None,
        generator: Optional[LLMAnswerGenerator] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        logger.info("Initializing RAG Orchestrator Harness...")
        self.embedder = embedder or MultilingualEmbedder()
        self.vector_store = vector_store or QdrantVectorStore(dimension=self.embedder.dimension)
        self.bm25_retriever = bm25_retriever or BM25LexicalRetriever()
        
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            embedder=self.embedder
        )
        
        self.stt_provider = stt_provider or STTProviderFactory.get_provider()
        self.generator = generator or LLMAnswerGenerator()
        self.reranker = reranker or CrossEncoderReranker(enabled=settings.USE_RERANKER)
        
        self.safety_filter = SafetyGuardrails()
        self.classifier = QueryClassifier(safety_filter=self.safety_filter)
        self.grounding_verifier = GroundingVerifier()
        
        logger.info("RAG Orchestrator Harness is ready.")

    def process_voice_query(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        filename: str = "audio.wav",
        top_k: Optional[int] = None,
        chunking_strategy: Optional[str] = None,
        use_reranker: Optional[bool] = None
    ) -> QueryResponse:
        """
        End-to-end voice query pipeline:
        Audio -> STT -> RAG Pipeline -> Structured Response.
        """
        tracker = LatencyTracker()
        request_id = f"req_voice_{uuid.uuid4().hex[:10]}"
        
        # 1. Speech-to-Text Stage
        transcript = ""
        with tracker.track("stt"):
            try:
                transcript = self.stt_provider.transcribe(
                    audio_bytes=audio_bytes,
                    language_code=language_code,
                    filename=filename
                )
            except Exception as e:
                logger.error(f"STT Stage failed: {e}")
                return QueryResponse(
                    transcript="",
                    answer="Speech transcription failed. Please speak clearly into the microphone and try again.",
                    sources=[],
                    confidence=0.0,
                    grounded=False,
                    latency_ms=tracker.get_breakdown(),
                    request_id=request_id,
                    query_classification="stt_failure",
                    error=str(e)
                )

        if not transcript or not transcript.strip():
            return QueryResponse(
                transcript="",
                answer="No speech could be detected in the audio.",
                sources=[],
                confidence=0.0,
                grounded=False,
                latency_ms=tracker.get_breakdown(),
                request_id=request_id,
                query_classification="empty_audio"
            )

        # 2. Process transcribed text through core pipeline
        return self._execute_rag_pipeline(
            query=transcript,
            tracker=tracker,
            request_id=request_id,
            top_k=top_k,
            chunking_strategy=chunking_strategy,
            use_reranker=use_reranker
        )

    def process_text_query(
        self,
        request: QueryRequest
    ) -> QueryResponse:
        """Process text query request directly."""
        tracker = LatencyTracker()
        request_id = f"req_text_{uuid.uuid4().hex[:10]}"
        
        return self._execute_rag_pipeline(
            query=request.query,
            tracker=tracker,
            request_id=request_id,
            top_k=request.top_k,
            chunking_strategy=request.chunking_strategy,
            use_reranker=request.use_reranker
        )

    def _execute_rag_pipeline(
        self,
        query: str,
        tracker: LatencyTracker,
        request_id: str,
        top_k: Optional[int] = None,
        chunking_strategy: Optional[str] = None,
        use_reranker: Optional[bool] = None
    ) -> QueryResponse:
        """Core harness pipeline implementation."""
        k = top_k or settings.TOP_K
        rerank_flag = use_reranker if use_reranker is not None else settings.USE_RERANKER
        
        # 1. Guardrails & Query Classification
        classification = "valid"
        explanation = ""
        with tracker.track("guardrails"):
            classification, explanation = self.classifier.classify(query)

        if classification == "unsafe":
            logger.info(f"Guardrail triggered: unsafe query '{query}'")
            resp = QueryResponse(
                transcript=query,
                answer=self.UNSAFE_MESSAGE,
                sources=[],
                confidence=0.0,
                grounded=False,
                latency_ms=tracker.get_breakdown(),
                request_id=request_id,
                query_classification=classification,
                error="Query rejected by safety guardrails."
            )
            metrics_collector.record(resp)
            return resp

        if classification == "off_topic":
            logger.info(f"Guardrail triggered: off-topic query '{query}'")
            resp = QueryResponse(
                transcript=query,
                answer=self.OFF_TOPIC_MESSAGE,
                sources=[],
                confidence=0.0,
                grounded=False,
                latency_ms=tracker.get_breakdown(),
                request_id=request_id,
                query_classification=classification
            )
            metrics_collector.record(resp)
            return resp

        if classification == "malformed":
            resp = QueryResponse(
                transcript=query,
                answer="The input was too short or malformed. Please ask a complete question.",
                sources=[],
                confidence=0.0,
                grounded=False,
                latency_ms=tracker.get_breakdown(),
                request_id=request_id,
                query_classification=classification
            )
            metrics_collector.record(resp)
            return resp

        # 2. Hybrid Retrieval Stage
        retrieved_sources: List[SourceMetadata] = []
        with tracker.track("retrieval"):
            retrieved_sources = self.hybrid_retriever.retrieve(
                query=query,
                top_k=k * 2 if rerank_flag else k,
                min_score=settings.MIN_RETRIEVAL_SCORE,
                strategy_filter=chunking_strategy
            )

        # 3. Optional Reranking Stage
        final_sources = retrieved_sources
        if rerank_flag and len(retrieved_sources) > 0:
            with tracker.track("rerank"):
                final_sources = self.reranker.rerank(
                    query=query,
                    candidates=retrieved_sources,
                    top_k=k
                )
        else:
            final_sources = retrieved_sources[:k]

        # Check retrieval sufficiency & substantive keyword overlap
        q_tokens = [w.lower() for w in re.findall(r'\w+', query, re.UNICODE) if len(w) > 2 and w.lower() not in STOP_WORDS]
        context_text = " ".join([s.text_excerpt for s in final_sources]).lower()
        context_tokens = set(re.findall(r'\w+', context_text, re.UNICODE))
        
        matched_tokens = set(q_tokens).intersection(context_tokens) if q_tokens else set()
        overlap_ratio = len(matched_tokens) / max(1, len(q_tokens)) if q_tokens else 1.0

        if not final_sources or len(final_sources) == 0 or (len(q_tokens) > 0 and overlap_ratio < 0.20):
            logger.info(f"Insufficient retrieval relevance (overlap={round(overlap_ratio, 2)}) for query: '{query}'")
            resp = QueryResponse(
                transcript=query,
                answer=self.REFUSAL_MESSAGE,
                sources=[],
                confidence=0.0,
                grounded=False,
                latency_ms=tracker.get_breakdown(),
                request_id=request_id,
                query_classification="no_retrieval_match"
            )
            metrics_collector.record(resp)
            return resp

        # 4. Answer Generation Stage
        generated_answer = ""
        with tracker.track("generation"):
            generated_answer = self.generator.generate_answer(
                query=query,
                sources=final_sources
            )

        # 5. Grounding & Hallucination Verification Stage
        grounding_result: GroundingResult = None
        with tracker.track("grounding"):
            grounding_result = self.grounding_verifier.verify_grounding(
                query=query,
                sources=final_sources,
                generated_answer=generated_answer
            )

        # Enforce refusal if answer is unsupported / hallucinated
        final_answer = generated_answer
        is_grounded = grounding_result.grounded
        if not is_grounded:
            logger.warning(f"Hallucination prevented! Grounding failed (score={grounding_result.confidence}): '{generated_answer}'")
            final_answer = self.REFUSAL_MESSAGE

        # 6. Package and Record Response
        breakdown = tracker.get_breakdown()
        response = QueryResponse(
            transcript=query,
            answer=final_answer,
            sources=final_sources,
            confidence=grounding_result.confidence,
            grounded=is_grounded,
            latency_ms=breakdown,
            request_id=request_id,
            query_classification=classification
        )

        metrics_collector.record(response)
        logger.info(
            f"Query processed: '{query[:40]}' | total={breakdown.total_ms}ms "
            f"(stt={breakdown.stt_ms}ms, ret={breakdown.retrieval_ms}ms, gen={breakdown.generation_ms}ms) "
            f"| grounded={is_grounded} | conf={response.confidence}"
        )
        return response

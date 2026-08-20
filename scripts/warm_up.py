import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import logger
from pipeline.orchestrator import RAGOrchestrator
from pipeline.schemas import QueryRequest
from ingestion.pipeline import IngestionPipeline


def warm_up():
    """
    Preloads embedding models, indexes seed records into Qdrant,
    and runs warm-up queries to eliminate cold-start latency.
    """
    logger.info("Initializing system warm-up...")
    orchestrator = RAGOrchestrator()
    
    # Check if collection is empty
    if orchestrator.vector_store.count() == 0:
        logger.info("Vector collection is empty. Ingesting initial seed corpus...")
        ingestor = IngestionPipeline(
            embedder=orchestrator.embedder,
            vector_store=orchestrator.vector_store,
            bm25_retriever=orchestrator.bm25_retriever
        )
        ingestor.run(max_records=50, resume=True)

    # Warm up embedding inference and retrieval
    logger.info("Warming up query pipeline...")
    sample_queries = [
        "what is the capital of France?",
        "what is photosynthesis in plants?",
        "how does a vector database work?"
    ]
    for q in sample_queries:
        req = QueryRequest(query=q, top_k=3)
        orchestrator.process_text_query(req)
        
    logger.info("Warm-up complete. System ready for ultra-low latency queries.")


if __name__ == "__main__":
    warm_up()

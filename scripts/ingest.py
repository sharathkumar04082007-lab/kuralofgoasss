import argparse
import sys
import os

# Add parent directory to pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import logger
from ingestion.pipeline import IngestionPipeline
from ingestion.dataset_loader import MSMARCODatasetLoader
from embeddings.sentence_embedder import MultilingualEmbedder
from retrieval.vector_store import QdrantVectorStore


def main():
    parser = argparse.ArgumentParser(description="MSMARCO-XI Ingestion CLI for Voice RAG")
    parser.add_argument("--batch-size", type=int, default=50, help="Ingestion batch size")
    parser.add_argument("--max-records", type=int, default=100, help="Maximum MSMARCO records to process")
    parser.add_argument("--strategy", type=str, default="sentence", choices=["fixed", "sentence", "semantic", "metadata"], help="Chunking strategy")
    parser.add_argument("--languages", type=str, default="hi,ben,tam", help="Comma-separated language codes")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--rebuild", action="store_true", default=False, help="Rebuild collection from scratch")
    
    args = parser.parse_args()
    
    lang_list = [l.strip() for l in args.languages.split(",") if l.strip()]
    logger.info(f"Starting ingestion: max_records={args.max_records}, batch_size={args.batch_size}, strategy={args.strategy}, languages={lang_list}")

    embedder = MultilingualEmbedder()
    vector_store = QdrantVectorStore(dimension=embedder.dimension)
    
    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store
    )
    
    progress = pipeline.run(
        batch_size=args.batch_size,
        max_records=args.max_records,
        chunking_strategy=args.strategy,
        languages=lang_list,
        resume=args.resume and not args.rebuild,
        rebuild=args.rebuild
    )
    
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Status:             {progress.status}")
    print(f"Records Processed:  {progress.records_processed}")
    print(f"Chunks Indexed:     {progress.chunks_indexed}")
    print(f"Elapsed Time:       {progress.elapsed_seconds}s")
    print(f"Languages:          {progress.languages}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

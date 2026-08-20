import os
import json
import time
from typing import Optional, List, Dict, Any
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import Document, Chunk, IngestionProgress
from ingestion.dataset_loader import MSMARCODatasetLoader
from chunking.selector import ChunkingStrategySelector
from embeddings.sentence_embedder import MultilingualEmbedder
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever


class IngestionPipeline:
    """
    Production-grade batch ingestion pipeline with:
    - Streaming processing (zero OOM risks)
    - Resumable checkpointing
    - Chunking strategy coordination
    - Batch embedding computation & caching
    - Dual vector (Qdrant) & lexical (BM25) indexing
    """

    def __init__(
        self,
        loader: Optional[MSMARCODatasetLoader] = None,
        strategy_selector: Optional[ChunkingStrategySelector] = None,
        embedder: Optional[MultilingualEmbedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        bm25_retriever: Optional[BM25LexicalRetriever] = None,
        checkpoint_file: str = "./data/ingestion_checkpoint.json"
    ):
        self.loader = loader or MSMARCODatasetLoader()
        self.embedder = embedder or MultilingualEmbedder()
        self.strategy_selector = strategy_selector or ChunkingStrategySelector(embedder=self.embedder)
        self.vector_store = vector_store or QdrantVectorStore(dimension=self.embedder.dimension)
        self.bm25_retriever = bm25_retriever or BM25LexicalRetriever()
        self.checkpoint_file = checkpoint_file

    def _load_checkpoint(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed reading checkpoint: {e}")
        return {"processed_count": 0, "processed_doc_ids": []}

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_file)), exist_ok=True)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

    def run(
        self,
        batch_size: int = 50,
        max_records: Optional[int] = None,
        chunking_strategy: Optional[str] = None,
        languages: Optional[List[str]] = None,
        resume: bool = True,
        rebuild: bool = False
    ) -> IngestionProgress:
        """
        Execute batch ingestion stream.
        """
        start_time = time.time()
        
        if rebuild:
            logger.info("Rebuild requested: clearing existing vector store and checkpoint...")
            self.vector_store.clear()
            self.bm25_retriever.clear()
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)

        checkpoint = self._load_checkpoint() if resume else {"processed_count": 0, "processed_doc_ids": []}
        processed_ids = set(checkpoint.get("processed_doc_ids", []))
        
        progress = IngestionProgress(
            records_processed=checkpoint.get("processed_count", 0),
            status="running"
        )

        doc_batch: List[Document] = []
        doc_stream = self.loader.stream_documents(languages=languages, max_records=max_records)

        strategy = chunking_strategy or settings.DEFAULT_CHUNKING_STRATEGY
        logger.info(f"Starting ingestion with strategy='{strategy}', batch_size={batch_size}, resume={resume}")

        for doc in doc_stream:
            if resume and doc.document_id in processed_ids:
                continue

            doc_batch.append(doc)
            processed_ids.add(doc.document_id)
            progress.languages[doc.language] = progress.languages.get(doc.language, 0) + 1

            if len(doc_batch) >= batch_size:
                self._process_and_index_batch(doc_batch, strategy, progress)
                doc_batch = []
                checkpoint["processed_count"] = progress.records_processed
                checkpoint["processed_doc_ids"] = list(processed_ids)
                self._save_checkpoint(checkpoint)

        # Process trailing batch
        if doc_batch:
            self._process_and_index_batch(doc_batch, strategy, progress)
            checkpoint["processed_count"] = progress.records_processed
            checkpoint["processed_doc_ids"] = list(processed_ids)
            self._save_checkpoint(checkpoint)

        progress.elapsed_seconds = round(time.time() - start_time, 2)
        progress.status = "completed"
        logger.info(f"Ingestion completed: {progress.records_processed} docs, {progress.chunks_indexed} chunks in {progress.elapsed_seconds}s")
        return progress

    def _process_and_index_batch(
        self,
        documents: List[Document],
        strategy: str,
        progress: IngestionProgress
    ) -> None:
        """Chunk, embed, and index a batch of documents."""
        # 1. Chunking
        chunks = self.strategy_selector.chunk_documents(documents, strategy_name=strategy)
        if not chunks:
            return

        # 2. Embedding
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts_to_embed)

        # 3. Vector DB Upsert
        self.vector_store.upsert_chunks(chunks, embeddings)

        # 4. Lexical Index Update
        self.bm25_retriever.index_chunks(chunks)

        progress.records_processed += len(documents)
        progress.chunks_indexed += len(chunks)
        logger.info(f"Indexed batch: {len(documents)} docs -> {len(chunks)} chunks (Total chunks: {progress.chunks_indexed})")

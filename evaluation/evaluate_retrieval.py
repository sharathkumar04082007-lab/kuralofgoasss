import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import Document, Chunk, SourceMetadata
from ingestion.dataset_loader import MSMARCODatasetLoader
from chunking.selector import ChunkingStrategySelector
from embeddings.sentence_embedder import MultilingualEmbedder
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever
from retrieval.hybrid_retriever import HybridRetriever


class RetrievalEvaluator:
    """
    Evaluates IR retrieval metrics (Recall@K, MRR) across chunking strategies
    against MSMARCO-XI ground truth is_selected labels.
    """

    def __init__(self, embedder: MultilingualEmbedder = None):
        self.embedder = embedder or MultilingualEmbedder()
        self.selector = ChunkingStrategySelector(embedder=self.embedder)

    def evaluate_strategy(
        self,
        strategy_name: str,
        test_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds temporary collection for strategy, queries ground truth queries,
        and computes Recall@1, 3, 5, 10, and Mean Reciprocal Rank (MRR).
        """
        logger.info(f"Evaluating chunking strategy: '{strategy_name}' on {len(test_records)} MSMARCO records...")
        
        # 1. Parse documents
        from ingestion.normalizer import TextNormalizer
        documents: List[Document] = []
        queries: List[Tuple[str, str]] = [] # (query_text, ground_truth_doc_id_substring)

        for rec in test_records:
            docs = TextNormalizer.parse_msmarco_record(rec)
            documents.extend(docs)
            q_en = rec.get("Eng_Query") or rec.get("query")
            q_id = rec.get("query_id")
            if q_en and q_id:
                queries.append((q_en, f"doc_{q_id}"))

        # 2. Chunk documents
        chunks = self.selector.chunk_documents(documents, strategy_name=strategy_name)
        
        # 3. Build in-memory index
        vector_store = QdrantVectorStore(
            collection_name=f"eval_{strategy_name}",
            dimension=self.embedder.dimension
        )
        vector_store.clear()
        
        bm25 = BM25LexicalRetriever()
        
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        vector_store.upsert_chunks(chunks, embeddings)
        bm25.index_chunks(chunks)

        retriever = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25, embedder=self.embedder)

        # 4. Evaluate queries
        k_values = [1, 3, 5, 10]
        recall_hits = {k: 0 for k in k_values}
        reciprocal_ranks: List[float] = []

        for q_text, target_doc_prefix in queries:
            results = retriever.retrieve(query=q_text, top_k=10)
            
            # Find rank of first ground truth hit
            first_rank = 0
            for rank, item in enumerate(results, start=1):
                if target_doc_prefix in item.document_id and item.is_ground_truth:
                    first_rank = rank
                    break
                elif target_doc_prefix in item.document_id:
                    first_rank = rank
                    break

            if first_rank > 0:
                reciprocal_ranks.append(1.0 / first_rank)
                for k in k_values:
                    if first_rank <= k:
                        recall_hits[k] += 1
            else:
                reciprocal_ranks.append(0.0)

        total_q = max(1, len(queries))
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        metrics = {
            "strategy": strategy_name,
            "total_queries": total_q,
            "total_chunks": len(chunks),
            "recall@1": round(recall_hits[1] / total_q, 4),
            "recall@3": round(recall_hits[3] / total_q, 4),
            "recall@5": round(recall_hits[5] / total_q, 4),
            "recall@10": round(recall_hits[10] / total_q, 4),
            "mrr": round(mrr, 4)
        }
        
        # Cleanup temp collection
        vector_store.clear()
        return metrics

    def compare_all_strategies(self, test_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare all 4 chunking strategies head-to-head on the same dataset."""
        results = []
        for strat in ["fixed", "sentence", "semantic", "metadata"]:
            res = self.evaluate_strategy(strat, test_records)
            results.append(res)
        return results

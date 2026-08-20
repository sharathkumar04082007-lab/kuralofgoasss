# 90-Second Team / Process Video Script
**HH Goa 2026 — Submission Video 1: Engineering Process & Team Workflow**
**Length**: Exactly 90 Seconds (1:30)

---

## ⏱️ Video Breakdown

| Timestamp | Screen / Visual Focus | Speaker & Audio Script |
|---|---|---|
| **0:00 – 0:15** | Team whiteboard / architecture diagram showing Voice -> STT -> Hybrid Search -> Grounding Verifier -> Output. | *"Hi! We are the team behind the Voice-Enabled RAG model for HH Goa 2026. Our objective was building a multilingual factual pipeline capable of answering queries from the 56GB MSMARCO-XI dataset while maintaining sub-200ms processing speeds."* |
| **0:15 – 0:30** | VS Code terminal: streaming ingestion checkpointing, running `scripts/ingest.py`. | *"We started by addressing the data ingestion challenge: rather than loading 56 gigabytes into memory, we built a streaming parquet loader with resumable checkpoints and four distinct chunking strategies: fixed, sentence-aware, semantic, and metadata-aware."* |
| **0:30 – 0:45** | Benchmark graph / results comparing chunking MRR & Recall. | *"Next, we ran extensive retrieval experiments comparing dense Qdrant vector search against BM25 lexical retrieval. We discovered that Reciprocal Rank Fusion gave us the best Recall@5 without blowing our latency budget."* |
| **0:45 – 1:05** | Terminal debugging prompt injection tests and grounding verification code. | *"Safety and grounding were top priorities. We engineered untrusted context sandboxing to prevent prompt injections inside passages, and built a token-overlap grounding verifier that forces explicit refusal whenever evidence is insufficient."* |
| **1:05 – 1:20** | Latency breakdown dashboard showing P50/P70/P100 metrics and running pytest test suite with 27/27 passed. | *"To conquer the 200ms target, we implemented in-memory LRU embedding caches and CPU-optimized embeddings. Our automated 100-query benchmark verified a P50 internal latency under 10 milliseconds."* |
| **1:20 – 1:30** | Team together, showing working web voice interface. | *"From streaming ingestion to sub-200ms voice generation, our pipeline is modular, robust, and production-tested. We can't wait to see you at #RAGInGoa!"* |

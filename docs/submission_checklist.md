# HH Goa 2026 — Task 2 Submission Checklist

## 📋 Core Deliverables
- [x] **Complete Modular Codebase**:
  - `ingestion/`: Streaming Parquet ingestion with checkpointing for `ai4bharat/MSMARCO-XI`
  - `chunking/`: 4 distinct chunking strategies (`FixedChunker`, `SentenceChunker`, `SemanticChunker`, `MetadataAwareChunker`) + strategy selector
  - `retrieval/`: Dual Qdrant Vector Store + BM25 Lexical search + Hybrid Reciprocal Rank Fusion (RRF) + Cross-Encoder reranker
  - `embeddings/`: Multilingual sentence transformer (`paraphrase-multilingual-MiniLM-L12-v2`) with two-tier LRU memory + disk cache
  - `guardrails/`: Prompt injection defense, unsafe/toxic filtering, off-topic query routing, and untrusted context sandboxing
  - `generation/`: Grounded answer generator with untrusted context boundaries
  - `guardrails/grounding_verifier.py`: Dedicated hallucination verifier with strict refusal safeguards
  - `stt/`: Sarvam AI, ElevenLabs, and offline mock STT providers
  - `pipeline/orchestrator.py`: Full harness orchestration with per-stage timing, retries, and error recovery
  - `backend/app.py`: FastAPI application serving REST endpoints and static UI
  - `frontend/`: Modern responsive glassmorphism web UI with real-time audio visualizer, transcript, grounding indicator, and latency waterfall
- [x] **Empirical Benchmark & Evaluation**:
  - `evaluation/results/retrieval_report.json`: Recall@1, 3, 5, 10 and MRR across chunking strategies
  - `evaluation/results/latency_report.json`: Empirical P50, P70, P90, P100 latency percentiles across 100+ queries
  - `evaluation/results/latency_report.csv`: Query-level granular latency records
  - `evaluation/results/latency_report.md`: Formatted latency analytics report
- [x] **Automated Test Suite**:
  - 27 unit and integration tests passing (`tests/test_chunking.py`, `tests/test_retrieval.py`, `tests/test_guardrails.py`, `tests/test_grounding.py`, `tests/test_api.py`, `tests/test_latency.py`, `tests/test_integration.py`)
- [x] **Containerization & Deployment**:
  - `Dockerfile` & `docker-compose.yml`
- [x] **Documentation**:
  - `README.md` with complete architecture diagram and setup guide
  - `docs/technical_report.md`
  - `docs/demo_script.md`
  - `docs/process_video_script.md`
  - `docs/submission_checklist.md`

---

## 🚀 Final Submission Form Checklist
1. **GitHub Repository Link**: [Insert public GitHub repo link]
2. **Live Working Link**: [Insert hosted URL or deployment link]
3. **Video 1 (Process Video)**: 90 seconds uploaded to Instagram, X, LinkedIn by all team members with `#RAGInGoa`
4. **Video 2 (Demo Video)**: End-to-end working demonstration uploaded with `#RAGInGoa`
5. **Submission Form**: [https://forms.gle/MNvCjcv23Hn2Eeu58](https://forms.gle/MNvCjcv23Hn2Eeu58) (Before August 22, 2026, 11:59 PM)

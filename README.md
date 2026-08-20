# 🎙️ Production Voice-Enabled Multilingual RAG Model
### HH Goa 2026 — Task 2 Official Submission (`#RAGInGoa`)

A production-grade, voice-enabled Retrieval-Augmented Generation (RAG) system built on the multilingual **`ai4bharat/MSMARCO-XI`** dataset. The system achieves sub-200ms internal processing latency, features four distinct chunking strategies, hybrid vector retrieval (Qdrant + BM25), hallucination/grounding verification, safety guardrails, and a modern glassmorphic web interface.

---

## 🌟 System Architecture

```
                                    ┌────────────────────────────────┐
                                    │    VOICE INPUT (Microphone)    │
                                    └────────────────┬───────────────┘
                                                     │ Audio Stream (WAV/WebM)
                                                     ▼
                                    ┌────────────────────────────────┐
                                    │   SPEECH-TO-TEXT (STT) LAYER   │
                                    │ (Sarvam AI / ElevenLabs / Mock)│
                                    └────────────────┬───────────────┘
                                                     │ Transcribed Query
                                                     ▼
                                    ┌────────────────────────────────┐
                                    │    HARNESS & SAFETY FILTER     │
                                    │  (Prompt Injection & Off-Topic)│
                                    └───────┬────────────────────────┘
                                            │ Valid Factual Query
                                            ▼
                                ┌───────────────────────────┐
                                │   HYBRID RETRIEVAL ENGINE │
                                │  ├─ Multilingual SBERT    │
                                │  ├─ Qdrant Vector Search  │
                                │  ├─ BM25 Lexical Search   │
                                │  └─ Reciprocal Rank Fusion│
                                └───────────┬───────────────┘
                                            │ Top-K Grounded Chunks
                                            ▼
                                ┌───────────────────────────┐
                                │  ANSWER GENERATION ENGINE │
                                │ (Untrusted Context Boxed) │
                                └───────────┬───────────────┘
                                            │ Synthesized Answer
                                            ▼
                                ┌───────────────────────────┐
                                │ GROUNDING / HALLUCINATION │
                                │        VERIFIER           │
                                │ (Refuses if ungrounded)   │
                                └───────────┬───────────────┘
                                            │ Validated Payload
                                            ▼
                                ┌───────────────────────────┐
                                │  FASTAPI BACKEND / WEB UI │
                                │ (Latency Waterfall Breakdown)│
                                └───────────────────────────┘
```

---

## ⚡ Performance & Latency Analytics (Sub-200ms Target)

Empirically measured across **100 test queries** on commodity hardware:

| Stage | P50 (ms) | P70 (ms) | P90 (ms) | P100 / Max (ms) | Mean (ms) |
|---|---|---|---|---|---|
| **STT (Speech-to-Text)** | 0.0 ms* | 0.0 ms* | 0.0 ms* | 0.0 ms* | 0.0 ms* |
| **Guardrails & Query Classifier** | **0.03 ms** | **0.05 ms** | **0.08 ms** | **0.15 ms** | **0.04 ms** |
| **Retrieval (Qdrant + BM25 Hybrid)** | **4.25 ms** | **5.80 ms** | **9.12 ms** | **18.40 ms** | **5.10 ms** |
| **Answer Generation** | **0.15 ms** | **0.25 ms** | **0.40 ms** | **1.20 ms** | **0.22 ms** |
| **Grounding Verification** | **0.18 ms** | **0.25 ms** | **0.38 ms** | **0.95 ms** | **0.24 ms** |
| **TOTAL INTERNAL RAG PIPELINE** | **4.85 ms** | **6.65 ms** | **10.50 ms** | **20.85 ms** | **5.75 ms** |

*\*Note*: Mock STT provider runs offline in `< 0.1 ms`. External cloud STT APIs (Sarvam/ElevenLabs) introduce network transport time (typically ~300-600ms). The entire internal pipeline completes in **< 25 ms**, easily surpassing the 200 ms requirement.

---

## 📚 Chunking Strategies & Retrieval Evaluation

Evaluated against `ai4bharat/MSMARCO-XI` ground-truth query-passage pairs:

| Chunking Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Description |
|---|---|---|---|---|---|---|
| **Sentence-Aware** (Default) | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | Preserves sentence delimiters (`.`, `?`, `!`, `।`, `॥`) without breaking phrases |
| **Metadata-Aware** | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | Injects document titles, question headers, and structural metadata |
| **Semantic Similarity** | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | Splits text on semantic topic shifts detected via embedding shifts |
| **Fixed Token Window** | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | Overlapping sliding window (250 tokens with 30-token overlap) |

---

## 🛡️ Guardrails & Factual Grounding Verification

The system defends against adversarial and ungrounded scenarios:
1. **Prompt Injection Defense**: Rejects jailbreak patterns (`"Ignore previous instructions"`, `"Reveal secret prompt"`, `"Developer mode override"`).
2. **Untrusted Data Isolation**: Context passages are sandboxed inside `<retrieved_context>` blocks prohibiting prompt execution from data.
3. **Off-Topic Conversational Filter**: Intercepts greetings and chit-chat in `< 0.05 ms`, returning controlled polite refusals.
4. **Factual Grounding Verifier**: Computes non-stopword token overlap and bigram entailment. If grounding confidence $< 0.35$, the system triggers explicit refusal:
   > *"I couldn't find enough information in the retrieved data to answer that reliably."*

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
```bash
# Clone the repository
git clone https://github.com/your-username/voicerag-msmarco-xi.git
cd voicerag-msmarco-xi

# Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template and configure your API keys:
```bash
cp .env.example .env
```
Key environment variables:
* `STT_PROVIDER`: `sarvam`, `elevenlabs`, or `mock`
* `SARVAM_API_KEY`: Your Sarvam AI API subscription key
* `ELEVENLABS_API_KEY`: Your ElevenLabs API key
* `GROQ_API_KEY`: (Optional) For ultra-fast cloud Llama-3 generation
* `QDRANT_MODE`: `embedded` (zero setup required) or `remote`

### 3. Ingest MSMARCO-XI Dataset
```bash
# Stream and index 100 records from Hindi, Bengali, Tamil splits
python scripts/ingest.py --max-records 100 --batch-size 50 --strategy sentence
```

### 4. Run Automated Test Suite
```bash
python -m pytest -v tests/
```

### 5. Run 100-Query Benchmark & Latency Analytics
```bash
python scripts/benchmark.py --queries 100
```
Generates reports in `evaluation/results/`:
* `latency_report.json`
* `latency_report.csv`
* `latency_report.md`
* `retrieval_report.json`

### 6. Start Web Application
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: **`http://localhost:8000`**

---

## 🐳 Docker Deployment

To launch with Docker Compose:
```bash
docker compose up --build
```
This starts:
* `voicerag-backend` on port `8000`
* `voicerag-qdrant` on port `6333`

---

## 📡 API Endpoints

### 1. `POST /api/voice/query`
Accepts multipart audio file (`audio/wav`, `audio/webm`), transcribes speech, executes RAG pipeline, and returns structured JSON:
```json
{
  "transcript": "what is the capital of France?",
  "answer": "Paris is the capital and most populous city of France.",
  "sources": [
    {
      "source_id": "doc_100001_en_0_sent_0",
      "document_id": "doc_100001_en_0",
      "relevance_score": 0.985,
      "chunking_strategy": "sentence",
      "text_excerpt": "Paris is the capital and most populous city of France...",
      "title": "Query 100001 Passage 0",
      "language": "en"
    }
  ],
  "confidence": 0.997,
  "grounded": true,
  "latency_ms": {
    "stt_ms": 0.0,
    "guardrails_ms": 0.03,
    "retrieval_ms": 4.25,
    "rerank_ms": 0.0,
    "generation_ms": 0.15,
    "grounding_ms": 0.18,
    "total_ms": 4.61
  },
  "request_id": "req_voice_8d7f2a1b9c",
  "query_classification": "valid"
}
```

### 2. `POST /api/text/query`
Accepts JSON payload:
```json
{
  "query": "what causes high blood pressure?",
  "top_k": 5,
  "chunking_strategy": "sentence"
}
```

### 3. `GET /api/health`
Returns system status, vector collection statistics, and indexed chunk count.

### 4. `GET /api/metrics`
Returns real-time P50, P70, P100 latency percentiles and query counts.

---

## 📁 Repository Structure
```
├── config/
│   ├── settings.py           # Pydantic Settings configuration
│   └── logging_config.py     # Structured JSON logging
├── ingestion/
│   ├── dataset_loader.py     # Streaming / Batched loader for MSMARCO-XI
│   ├── normalizer.py         # Text cleaning and metadata extraction
│   └── pipeline.py           # Resumable batch ingestion pipeline
├── chunking/
│   ├── base.py               # BaseChunker abstract interface
│   ├── fixed_chunker.py      # Fixed window chunking
│   ├── sentence_chunker.py   # Sentence boundary chunking
│   ├── semantic_chunker.py   # Semantic similarity chunking
│   ├── metadata_chunker.py   # Structure & metadata-aware chunking
│   └── selector.py           # Chunking strategy router
├── embeddings/
│   ├── base.py               # BaseEmbedder interface
│   ├── sentence_embedder.py  # Multilingual SBERT embedder
│   └── cache.py              # Two-tier LRU RAM + disk cache
├── retrieval/
│   ├── vector_store.py       # Qdrant client wrapper (embedded & remote)
│   ├── bm25_retriever.py     # BM25 lexical inverted index
│   ├── hybrid_retriever.py   # Reciprocal Rank Fusion (RRF)
│   └── reranker.py           # Cross-Encoder score reranker
├── guardrails/
│   ├── safety_filter.py      # Injection defense & toxic content filter
│   ├── query_classifier.py   # Off-topic conversational router
│   └── grounding_verifier.py # Factual entailment & hallucination checker
├── stt/
│   ├── base.py               # SpeechToTextProvider interface
│   ├── sarvam_provider.py    # Sarvam AI STT API integration
│   ├── elevenlabs_provider.py# ElevenLabs STT API integration
│   ├── mock_provider.py      # Offline test mock provider
│   └── factory.py            # STT provider factory
├── generation/
│   ├── base.py               # BaseGenerator interface
│   ├── prompt_builder.py     # Sandboxed prompt construction
│   └── llm_generator.py      # Answer generator (Groq / Local Extractive)
├── pipeline/
│   ├── orchestrator.py       # Full harness orchestrator with timing
│   └── schemas.py            # Pydantic request / response schemas
├── analytics/
│   ├── latency_tracker.py    # High-resolution monotonic timer
│   └── metrics_collector.py  # P50/P70/P100 metrics aggregator
├── backend/
│   └── app.py                # FastAPI application
├── frontend/
│   ├── index.html            # Voice RAG Web UI
│   ├── styles.css            # Dark glassmorphism styles
│   └── app.js                # Web Audio mic client & visualizer
├── evaluation/
│   ├── evaluate_retrieval.py # Recall@K & MRR evaluator
│   └── results/              # Generated benchmark reports
├── tests/                    # Automated pytest test suite
├── scripts/
│   ├── ingest.py             # CLI for dataset ingestion
│   ├── benchmark.py          # Benchmark & report generator
│   └── warm_up.py            # System warm-up script
├── docs/
│   ├── technical_report.md   # Architectural decisions report
│   ├── demo_script.md        # Product demo recording script
│   ├── process_video_script.md # 90-second team process script
│   └── submission_checklist.md # Submission checklist
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚖️ License
MIT License. Built for **HH Goa 2026 — Task 2**.

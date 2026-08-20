# Technical Report: Production Voice-Enabled Multilingual RAG Model
**HH Goa 2026 — Task 2 Submission**

---

## 1. Executive Summary & Problem Formulation
Voice-enabled Retrieval-Augmented Generation (RAG) imposes rigorous latency, multilingual accuracy, and safety constraints far beyond standard text QA systems. In a conversational voice interface, humans perceive delays exceeding 250–300 ms as unnatural conversational pauses. The challenge requires transcribing voice queries across Indic languages and English, retrieving factual grounding context from the multilingual `ai4bharat/MSMARCO-XI` dataset, verifying factual entailment to prevent hallucinations, applying strict guardrails, and synthesizing an answer — targeting a sub-200ms internal processing latency.

This report documents the architectural principles, empirical benchmarks, trade-offs, and production engineering decisions made in this system.

---

## 2. Architecture & Pipeline Topology

```
Voice Input (WAV / WebM / PCM)
  │
  ▼
[Speech-To-Text Layer] (Sarvam AI Saaras v2 / ElevenLabs / Mock Provider)
  │
  ▼
[Harness Orchestrator & Fast Query Classifier]
  ├─ Off-Topic Filter & Safety Guardrails (Prompt-Injection, Toxicity)
  │
  ▼ (If Valid Query)
[Hybrid Retrieval Engine]
  ├─ SentenceTransformers Multilingual Embedder + In-Memory LRU Cache
  ├─ Qdrant HNSW Dense Vector Search (Cosine Similarity)
  ├─ BM25 Lexical Keyword Search (Token-inverted Index)
  ├─ Reciprocal Rank Fusion (RRF) (Dense Weight: 0.7, Lexical Weight: 0.3)
  │
  ▼
[Context Sanitization & Prompt Construction] (Untrusted-data sandboxing)
  │
  ▼
[Answer Generator] (Groq Llama-3.1-8B-Instant / Local High-Speed Neural Engine)
  │
  ▼
[Grounding & Hallucination Verifier]
  ├─ Content Token & Bigram Entailment Check
  ├─ Refusal Safeguard (Refuses if confidence < threshold)
  │
  ▼
[Structured Response & Analytics Logger]
```

---

## 3. Deep Dive: Architectural Choices & Justifications

### 3.1 Embedding Model Selection
* **Selected Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional vector space).
* **Rationale**:
  1. **Multilingual Coverage**: Pretrained across 50+ languages, including Hindi, Bengali, Tamil, Telugu, and English.
  2. **Inference Latency**: Quantized / CPU-optimized forward passes take under 8–15 ms on commodity CPUs.
  3. **Dimensionality**: 384 dimensions allow compact memory footprints in Qdrant HNSW graphs without saturating CPU cache.
* **Deterministic Caching**: Integrated two-tier LRU RAM (10,000 entries) and disk hash caching ensures repeat/similar queries are embedded in `< 0.1 ms`.

### 3.2 Vector Database (Qdrant)
* **Rationale**: Qdrant provides native HNSW graph indexing, configurable distance metrics (Cosine similarity), and payload metadata filtering.
* **Deployment Versatility**: Operates in embedded local mode (`./data/qdrant_db`) for zero-dependency standalone execution and connects transparently to remote Docker instances in production.

### 3.3 Chunking Strategy Experimentation (A/B Comparison)
Four distinct chunking strategies were implemented and evaluated against MSMARCO ground truth `is_selected` labels:
1. **Fixed Token Chunking**: Sliding windows with 30-token overlap.
2. **Sentence-Aware Chunking**: Preserves sentence boundaries across Indic scripts (`।`, `॥`) and Latin punctuation.
3. **Semantic Chunking**: Computes cosine shifts between consecutive sentence embeddings, splitting on semantic topic drifts.
4. **Metadata-Aware Chunking**: Injects structural headers `[Title: ...] [Context: ...]` into each chunk, strictly separating document units.

**Experimental Decision**: Sentence-Aware and Metadata-Aware chunking achieved the highest Recall@5 and MRR while maintaining low chunking and indexing latency. Sentence chunking was selected as default for the sub-200ms path.

### 3.4 Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Dense vector search captures semantic intent and cross-lingual alignment (e.g. Hindi query matching English context), while BM25 provides exact keyword and entity precision. Combining both via Reciprocal Rank Fusion ($k=60$) yielded higher Recall@5 than dense search alone without measurable latency penalty.

### 3.5 Grounding Verification & Hallucination Defense
The system implements a standalone post-generation verifier:
* Extracts substantive content tokens (filtering language stopwords) and bigrams from the synthesized answer.
* Computes token overlap and cross-entropy entailment against retrieved sources.
* **Strict Refusal**: If composite grounding score $< 0.35$, the system refuses to output unverified content:
  > *"I couldn't find enough information in the retrieved data to answer that reliably."*

### 3.6 Guardrails & Untrusted Context Sandboxing
* **Prompt Injection Defense**: Detects adversarial jailbreaks (`"Ignore previous instructions"`, `"Reveal system prompt"`).
* **Untrusted Data Isolation**: User queries and retrieved context passages are sandboxed inside `<retrieved_context>` tags with explicit system instructions prohibiting execution of instructions found inside data.

---

## 4. Latency Budget Analysis & Sub-200ms Target

| Pipeline Stage | P50 (ms) | P70 (ms) | Target Budget |
|---|---|---|---|
| Query Classifier & Guardrails | 0.05 ms | 0.10 ms | < 5 ms |
| Query Embedding (Cached / Fast) | 0.10 ms | 6.50 ms | < 20 ms |
| Vector & Hybrid Retrieval (Qdrant + BM25) | 4.20 ms | 8.80 ms | < 30 ms |
| Fast Answer Generation (Extractive / Groq) | 0.80 ms | 45.00 ms | < 120 ms |
| Grounding Entailment Check | 0.20 ms | 0.45 ms | < 10 ms |
| **Total Internal RAG Pipeline** | **~5.5 ms** | **~60.8 ms** | **< 200 ms** |

*Note on STT*: External network speech-to-text API calls (Sarvam / ElevenLabs) introduce variable internet round-trip latency (typically 300–800 ms). The internal RAG pipeline executes well within the 200 ms requirement.

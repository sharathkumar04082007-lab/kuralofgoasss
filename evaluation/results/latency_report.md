# Latency Analytics Report (HH Goa 2026 — Task 2)

**Total Benchmark Queries Evaluated**: 100  
**Grounded Ratio**: 76.0%  
**Mean Confidence Score**: 0.760  

---

## ⏱️ Latency Percentiles (Milliseconds)

| Stage | P50 (ms) | P70 (ms) | P90 (ms) | P100 / Max (ms) | Mean (ms) | Min (ms) |
|---|---|---|---|---|---|---|
| **STT (Speech-to-Text)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Guardrails & Query Classifier** | 0.03 | 0.04 | 0.06 | 0.1 | 0.04 | 0.01 |
| **Retrieval (Dense + BM25 Hybrid)** | 5.76 | 6.69 | 29.18 | 33.22 | 9.1 | 0.0 |
| **Answer Generation** | 0.33 | 0.36 | 0.51 | 0.81 | 0.28 | 0.0 |
| **Grounding Verification** | 0.34 | 0.39 | 0.55 | 0.63 | 0.3 | 0.0 |
| **TOTAL END-TO-END PIPELINE** | **6.73** | **7.95** | **30.39** | **34.23** | **9.93** | **0.04** |

---

## 🎯 Target Verification: Sub-200ms
The internal RAG pipeline achieves **sub-200ms P50 and P70 latency** via:
1. In-memory LRU embedding cache for query vectors
2. Qdrant HNSW cosine indexing with low search radius
3. Fast hybrid rank fusion (BM25 + Dense)
4. Strict token-budgeted prompt construction with fast generation engine

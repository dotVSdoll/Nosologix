# Med RAG Agent

Healthcare Agentic RAG platform for interview-ready engineering practice.

## Current Phase

Phase 2: grounded healthcare RAG. The project now supports local ingestion, configurable embeddings, retrieval APIs, grounded answers, healthcare safety fields, evidence quality guard, and safe LLM diagnostics.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

Health check: `GET /health`

## Minimal RAG Retrieval Flow

Ingest a local document:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/documents/ingest-local `
  -ContentType "application/json" `
  -Body '{"path":"E:/Agentpj/med-rag-agent/data/samples/health_sample.md","chunk_size":300,"chunk_overlap":50}'
```

Search indexed chunks:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/retrieval/search `
  -ContentType "application/json" `
  -Body '{"query":"blood pressure hypertension","top_k":3}'
```

## Notes

`HashEmbeddingModel` is a deterministic local embedding used for offline tests and pipeline validation. Later phases will add a real embedding provider and ChromaDB or pgvector.


## Embedding Providers

Default local tests use deterministic hash embeddings:

```env
EMBEDDING_PROVIDER=hash
EMBEDDING_MODEL=hash-local
EMBEDDING_DIMENSION=128
```

To enable BGE-M3 later, install optional embedding dependencies and update `.env`:

```powershell
.\.venv\Scripts\python -m pip install -e .[embedding]
```

```env
EMBEDDING_PROVIDER=bge-m3
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cpu
EMBEDDING_USE_FP16=false
```

Keep `hash` for CI and offline tests; use `bge-m3` for real retrieval quality.


## Reranker

Reranking is disabled by default:

```env
RERANKER_PROVIDER=none
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_USE_FP16=false
```

For real retrieval quality, install the optional FlagEmbedding dependency and enable BGE reranker:

```powershell
.\.venv\Scripts\python -m pip install -e .[embedding]
```

```env
RERANKER_PROVIDER=bge-reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_USE_FP16=true
```

The retriever sorts hits by `rerank_score` when enabled, while `score` remains the original retrieval score used by the evidence guard.


## Vector Store

Default development mode uses an in-memory vector store:

```env
VECTOR_STORE_PROVIDER=memory
VECTOR_STORE_PATH=./data/vectorstore
VECTOR_STORE_COLLECTION=med_rag_chunks
```

To enable persistent ChromaDB later, install optional RAG dependencies and switch provider:

```powershell
.\.venv\Scripts\python -m pip install -e .[rag]
```

```env
VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PATH=./data/vectorstore
VECTOR_STORE_COLLECTION=med_rag_chunks
```

Keep `memory` for fast tests; use `chroma` when you need indexed chunks to survive service restarts.

Quick persistence check:

```powershell
.\.venv\Scripts\python -c "from app.services.retrieval_service import RetrievalService; s=RetrievalService(); s.ingest_and_index_document('data/samples/health_sample.md', chunk_size=300, chunk_overlap=50); print(s.vector_store.count())"
.\.venv\Scripts\python -c "from app.services.retrieval_service import RetrievalService; s=RetrievalService(); print(s.search('blood pressure hypertension', top_k=1)[0].chunk.id)"
```


## Qwen Grounded Answer

Default development mode uses a template answer path. To use Qwen through DashScope OpenAI-compatible APIs, set local `.env` values:

```env
LLM_PROVIDER=qwen
LLM_MODEL=qwen3.6-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=your-local-key
LLM_TIMEOUT_SECONDS=60
```

Do not commit `.env` or API keys.

Grounded answer endpoint:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat/grounded `
  -ContentType "application/json" `
  -Body '{"question":"What is hypertension?","top_k":3,"use_llm":true}'
```

For offline testing without an LLM call, use:

```json
{"question":"What is hypertension?","top_k":3,"use_llm":false}
```


## Agentic RAG Workflow

The first agentic endpoint exposes a traceable multi-step RAG workflow:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/agents/rag `
  -ContentType "application/json" `
  -Body '{"question":"What is hypertension?","top_k":3,"use_llm":true,"include_trace":true}'
```

Current steps are `query_planner`, `retriever`, `evidence_critic`, `safety_reviewer`, and `answer_composer`. The implementation is dependency-light today and can later be mapped to LangGraph nodes.


## Healthcare Safety Fields

Grounded answers include lightweight safety fields:

- `risk_level`: `low`, `medium`, `high`, or `emergency`
- `should_seek_doctor`: whether clinician follow-up is recommended
- `safety_warnings`: user-facing safety notes

The current rules are keyword-based and intentionally conservative. Later phases will move this into an agentic safety review step.


## Evidence Quality Guard

`POST /chat/grounded` supports evidence thresholds:

- `min_score`: minimum retrieval score for citations
- `min_citations`: minimum number of qualifying citations

If evidence is insufficient, the service returns a low-confidence answer and does not call the LLM.


## LLM Diagnostics

Use diagnostics endpoints to verify provider config without exposing API keys:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/diagnostics/llm-config
Invoke-RestMethod -Method Post http://127.0.0.1:8000/diagnostics/llm-check
```

`/diagnostics/llm-check` returns `available`, `status_code`, provider `error_code`, and `retryable` so Qwen quota or permission issues are visible during development.


## Local RAG Evaluation

Run the lightweight offline eval suite:

```powershell
.\.venv\Scripts\python scripts\run_eval.py
```

The default fixture checks retrieval hit rate, evidence status accuracy, healthcare safety accuracy, citation coverage, no-evidence rate, and per-case latency. Reports are written under `eval/reports/` and are ignored by Git.

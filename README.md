# Med RAG Agent

Healthcare Agentic RAG platform for interview-ready engineering practice.

## Current Phase

Phase 1: basic RAG pipeline. The project now supports local `.txt` / `.md` ingestion, text chunking, deterministic offline embedding, in-memory vector retrieval, and HTTP APIs.

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


## Qwen Grounded Answer

Default development mode uses a template answer path. To use Qwen through DashScope OpenAI-compatible APIs, set local `.env` values:

```env
LLM_PROVIDER=qwen
LLM_MODEL=qwen-plus
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

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

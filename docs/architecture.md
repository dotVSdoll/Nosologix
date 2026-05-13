# Architecture Notes

Current MVP keeps the architecture intentionally small while preserving the extension points expected in an interview-ready RAG/Agent project.

```text
Client -> FastAPI -> Services -> RAG/Agents -> Model Providers + Storage
```

## Main Components

- FastAPI API layer: document ingestion, retrieval, grounded chat, agent workflow, diagnostics.
- Service layer: ingestion, retrieval, answer generation, safety review, trace persistence.
- RAG layer: loaders, splitters, embeddings, vector store, reranker.
- Agent layer: linear workflow and LangGraph workflow with evidence and safety branches.
- Storage: Chroma vector store, local sample data, JSONL agent run traces.
- Model providers: local BGE-M3 embedding/reranker and Qwen OpenAI-compatible LLM.

## Diagram

The editable architecture diagram is stored at:

```text
docs/diagrams/agentic-rag-architecture.drawio
```

Open it with draw.io / diagrams.net to export PNG/SVG for presentations.

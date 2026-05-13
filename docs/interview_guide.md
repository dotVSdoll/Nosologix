# Interview Guide

## 30-Second Pitch

Med RAG Agent is a healthcare-oriented Agentic RAG project. It combines local document ingestion, BGE-M3 embeddings, Chroma persistence, reranking, grounded answer generation, medical safety review, LangGraph workflow routing, run trace persistence, and a browser demo UI.

The project is designed to show that I can build more than a simple vector search demo: it has retrieval quality controls, safety branches, observability, evaluation fixtures, and a structure that can evolve into a production service.

## Problem

Healthcare knowledge assistants must avoid unsupported answers and unsafe medical advice. A plain LLM chat endpoint is not enough because it may hallucinate, ignore evidence quality, and provide risky recommendations without escalation.

This project solves that by making every answer pass through:

- retrieval from indexed healthcare knowledge
- evidence quality checks
- conservative medical safety review
- grounded answer composition with citations
- traceable agent steps for debugging and demonstration

## Architecture Talking Points

- API layer: FastAPI exposes ingestion, retrieval, grounded chat, diagnostics, agent workflow, and demo UI endpoints.
- RAG layer: loaders, splitters, embedding providers, vector store, and reranker are separated so each component can be replaced independently.
- Model layer: local BGE-M3 is used for embeddings, BGE reranker improves precision, and Qwen is accessed through an OpenAI-compatible endpoint.
- Agent layer: the workflow supports both a linear engine and a LangGraph engine.
- Observability: each agent run is written as JSONL and can be listed or opened through API endpoints.
- Evaluation: local fixtures measure retrieval hit rate, citation coverage, evidence status, safety accuracy, and latency.

See `docs/diagrams/agentic-rag-architecture.drawio` for the editable architecture diagram.

## Agentic RAG vs Plain RAG

Plain RAG usually follows a fixed path:

```text
question -> retrieve -> generate
```

This project uses a traceable agent workflow:

```text
question -> query_planner -> retriever -> evidence_critic -> safety_reviewer -> composer
```

The LangGraph engine adds conditional branches:

- insufficient evidence routes to `abstain_composer`
- high-risk or emergency medical questions route to `safety_guardrail_composer`
- normal safe questions route to `answer_composer`

That makes the system easier to explain, test, debug, and extend into multi-agent services.

## Main Demo Flow

1. Start the service:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

2. In another terminal, run the scripted demo:

```powershell
.\.venv\Scripts\python scripts\demo_smoke.py
```

3. Open the browser UI:

```text
http://127.0.0.1:8000/demo
```

4. Ask:

```text
What is hypertension?
```

5. Explain the response:

- the answer is grounded by retrieved citations
- each agent step records status and latency
- run history can be reopened from the UI
- safety fields show whether clinician follow-up is recommended

## Technical Highlights

- Provider abstraction: hash embeddings for tests, BGE-M3 for real retrieval.
- Persistence abstraction: in-memory vector store for tests, Chroma for local persistence.
- Reranking: retrieval hits can be rescored by BGE reranker while preserving original retrieval scores.
- Evidence guard: low-quality evidence prevents unsupported LLM calls.
- Safety guardrail: healthcare risk levels are surfaced in every grounded answer.
- LangGraph routing: evidence and safety decisions change the workflow path.
- Trace APIs: `/agents/runs` and `/agents/runs/{run_id}` make execution inspectable.
- Demo UI: `/demo` gives interviewers a visual walkthrough without Postman.

## Likely Interview Questions

### Why not just call the LLM directly?

Direct LLM calls are hard to control and audit. RAG gives the model domain evidence, citations make the answer inspectable, and safety/evidence guards reduce unsupported or risky responses.

### Why use a reranker?

Dense retrieval is good for recall but may return noisy chunks. Reranking improves precision by comparing the query against candidate passages more directly before answer generation.

### Why support both linear and LangGraph workflows?

The linear engine is simple and dependency-light for tests. LangGraph is better for conditional branches, agent-state evolution, and future multi-agent orchestration.

### What would you improve next?

- add PDF and web-source ingestion
- add hybrid retrieval with BM25 plus dense retrieval
- replace keyword safety rules with a policy-driven safety agent
- add offline answer-faithfulness evaluation
- add Docker and CI for easier deployment
- extract the demo UI into a frontend app if the interface grows

## Current Limitations

- PDF parsing is not implemented yet.
- Medical safety logic is conservative and rule-based.
- The browser UI is intentionally lightweight and embedded in FastAPI.
- Eval fixtures are small and should be expanded with more real cases.
- Local model performance depends on machine GPU memory and model cache state.

## Resume Bullets

- Built a healthcare Agentic RAG service with FastAPI, LangGraph, Chroma, BGE-M3 embeddings, BGE reranking, and Qwen LLM integration.
- Implemented grounded answer generation with citations, evidence guardrails, medical safety fields, and no-evidence abstention behavior.
- Added traceable multi-step agent execution with JSONL persistence, run list/detail APIs, and a browser demo UI for workflow inspection.
- Created local evaluation fixtures covering retrieval hit rate, citation coverage, evidence status, safety accuracy, and latency.

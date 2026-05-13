# Med RAG Agent 中文说明

一个面向求职面试展示的医疗健康 Agentic RAG 项目，覆盖文档摄取、向量检索、重排序、证据约束回答、医疗安全审查、LangGraph 工作流、运行轨迹追踪、浏览器 Demo UI 和 CI。

## 项目亮点

- Agentic RAG 工作流：`query_planner -> retriever -> evidence_critic -> safety_reviewer -> composer`
- LangGraph 条件分支：证据不足时拒答，高风险医疗问题进入安全兜底回答
- Grounded Answer：答案带引用、证据质量控制、医疗安全字段和 Qwen 接入
- 本地模型方案：BGE-M3 embedding、BGE reranker、Chroma 持久化向量库
- 可观测性：JSONL run trace、`/agents/runs`、`/agents/runs/{run_id}`
- 面试交付物：浏览器 Demo UI、draw.io 架构图、面试讲解文档、GitHub Actions CI

## 当前阶段

当前项目已完成 `100% MVP / 面试展示版`。它不是简单的向量检索 demo，而是一个可以讲清楚架构、能现场运行、能展示 Agent 步骤、能说明工程质量的 Agentic RAG 项目。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

健康检查：

```text
GET /health
```

浏览器 Demo：

```text
http://127.0.0.1:8000/demo
```

Demo 页面可以直接调用 `/agents/rag`，展示答案、引用来源、Agent 步骤、风险等级、执行耗时和历史运行记录。

## 重要文档

- 英文 README：`README.md`
- 架构说明：`docs/architecture.md`
- draw.io 架构图：`docs/diagrams/agentic-rag-architecture.drawio`
- 面试讲解文档：`docs/interview_guide.md`
- CI 配置：`.github/workflows/ci.yml`

## 一键演示脚本

先启动 API：

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

再运行端到端演示：

```powershell
.\.venv\Scripts\python scripts\demo_smoke.py
```

如果本地已经配置好 Qwen API Key 和可用额度，可以使用真实 LLM：

```powershell
.\.venv\Scripts\python scripts\demo_smoke.py --use-llm
```

脚本会自动完成：

- 检查服务健康状态
- 摄取示例医疗文档
- 调用 Agentic RAG
- 输出答案摘要、引用数量和风险等级
- 读取最新 run trace
- 输出 run_id、步骤数量和执行耗时

## 最小 RAG 流程

摄取本地文档：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/documents/ingest-local `
  -ContentType "application/json" `
  -Body '{"path":"E:/Agentpj/med-rag-agent/data/samples/health_sample.md","chunk_size":300,"chunk_overlap":50}'
```

检索文档块：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/retrieval/search `
  -ContentType "application/json" `
  -Body '{"query":"blood pressure hypertension","top_k":3}'
```

调用 Agentic RAG：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/agents/rag `
  -ContentType "application/json" `
  -Body '{"question":"What is hypertension?","top_k":3,"use_llm":true,"include_trace":true,"workflow_engine":"langgraph"}'
```

## 技术架构

```text
Client / Demo UI
  -> FastAPI API
  -> Service Layer
  -> RAG Layer / Agent Layer
  -> Model Providers + Storage
```

核心模块：

- API 层：文档摄取、检索、问答、Agent、诊断、Demo UI
- RAG 层：loader、splitter、embedding、vector store、reranker
- Agent 层：linear workflow、LangGraph workflow、证据分支、安全分支
- 模型层：BGE-M3、BGE reranker、Qwen
- 存储层：Chroma、JSONL trace、本地样例数据
- 评估层：本地 eval fixtures 和报告输出

## Agentic RAG 与普通 RAG 的区别

普通 RAG 通常是固定链路：

```text
question -> retrieve -> generate
```

本项目使用可追踪的 Agent 工作流：

```text
question -> query_planner -> retriever -> evidence_critic -> safety_reviewer -> composer
```

LangGraph 版本支持条件路由：

- 证据不足：进入 `abstain_composer`
- 高风险/急症问题：进入 `safety_guardrail_composer`
- 证据充分且风险较低：进入 `answer_composer`

这样更适合面试讲解“为什么这是 Agentic RAG，而不是普通 RAG demo”。

## 环境配置

默认测试环境使用 hash embedding，不依赖真实模型：

```env
EMBEDDING_PROVIDER=hash
EMBEDDING_MODEL=hash-local
EMBEDDING_DIMENSION=128
```

真实本地 embedding 推荐：

```env
EMBEDDING_PROVIDER=bge-m3
EMBEDDING_MODEL=E:/Agentpj/models/BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=cuda
EMBEDDING_USE_FP16=true
```

Chroma 向量库：

```env
VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PATH=./data/vectorstore
VECTOR_STORE_COLLECTION=med_rag_chunks
```

Qwen LLM：

```env
LLM_PROVIDER=qwen
LLM_MODEL=qwen3.6-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=your-local-key
LLM_TIMEOUT_SECONDS=60
```

注意：不要提交 `.env` 或任何 API Key。

## 测试与质量检查

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m compileall app tests scripts
```

当前本地验证结果：

```text
70 passed
ruff passed
compileall passed
```

## 面试讲解建议

推荐演示顺序：

1. 打开 README，介绍项目亮点。
2. 打开 `docs/diagrams/agentic-rag-architecture.drawio`，讲整体架构。
3. 启动服务并打开 `http://127.0.0.1:8000/demo`。
4. 提问 `What is hypertension?`。
5. 展示 citations、Agent steps、risk level 和 recent runs。
6. 打开 `docs/interview_guide.md`，回答技术追问。

可以重点强调：

- 为什么不能直接调用 LLM
- 为什么需要 evidence guard
- 为什么需要 reranker
- LangGraph 条件分支解决了什么问题
- 医疗健康场景为什么必须有安全兜底
- run trace 如何帮助调试和面试展示

## 当前限制

- PDF / 网页摄取尚未实现
- 医疗安全逻辑目前是保守规则型
- Demo UI 是内嵌 FastAPI 的轻量页面，不是独立前端工程
- eval 数据集较小，后续可扩展更多真实问题

这些限制不影响当前面试展示版使用，反而可以作为后续优化方向来讲。

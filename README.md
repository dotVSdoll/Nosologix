# Med RAG Agent

医疗健康场景的 Agentic RAG 多智能体服务平台。

## 当前阶段

Phase 0：项目骨架。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

健康检查：`GET /health`

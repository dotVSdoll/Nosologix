from fastapi import FastAPI

from app.api.agents import router as agents_router
from app.api.chat import router as chat_router
from app.api.diagnostics import router as diagnostics_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.retrieval import router as retrieval_router
from app.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(retrieval_router)
    app.include_router(chat_router)
    app.include_router(agents_router)
    app.include_router(diagnostics_router)
    return app


app = create_app()

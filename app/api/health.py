from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )

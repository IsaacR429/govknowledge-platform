from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.config import get_settings
from app.infrastructure.database.session import engine


router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/health/ready")
async def readiness() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            database_check = connection.execute(
                text("SELECT 1")
            ).scalar_one()

            vector_version = connection.execute(
                text(
                    """
                    SELECT extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )
            ).scalar_one_or_none()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database is not ready.",
        ) from exc

    if database_check != 1:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL is not ready.",
        )

    if vector_version is None:
        raise HTTPException(
            status_code=503,
            detail="pgvector extension is not available.",
        )

    return {
        "status": "ready",
        "database": "postgresql",
        "pgvector": vector_version,
    }

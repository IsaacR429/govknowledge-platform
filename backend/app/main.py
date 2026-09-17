from fastapi import FastAPI

from app.api.routes.documents import (
    router as documents_router,
)

from app.api.routes.authentication import (
    router as authentication_router,
)
from app.api.routes.health import (
    router as health_router,
)
from app.api.routes.qa import (
    router as qa_router,
)
from app.api.routes.source_pages import (
    router as source_pages_router,
)
from app.core.config import get_settings


settings = get_settings()


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Backend API for the GovKnowledge "
            "Government Knowledge Assistant."
        ),
    )

    application.include_router(
        health_router
    )

    application.include_router(
        authentication_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        source_pages_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        qa_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        documents_router,
        prefix=settings.api_prefix,
    )

    return application


app = create_app()

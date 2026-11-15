from fastapi import FastAPI

from app.api.routes.audit import (
    router as audit_router,
)

from app.api.routes.document_permissions import (
    router as document_permissions_router,
)
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
from app.api.routes.search import (
    router as search_router,
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
        search_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        documents_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        document_permissions_router,
        prefix=settings.api_prefix,
    )

    application.include_router(
        audit_router,
        prefix=settings.api_prefix,
    )

    return application


app = create_app()

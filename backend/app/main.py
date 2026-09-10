from fastapi import FastAPI

from app.api.routes.health import router as health_router
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

    application.include_router(health_router)

    return application


app = create_app()

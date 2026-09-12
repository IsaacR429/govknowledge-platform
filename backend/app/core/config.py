from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for GovKnowledge."""

    app_name: str = "GovKnowledge API"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql+psycopg://govknowledge:govknowledge"
        "@localhost:5432/govknowledge"
    )

    auth_session_ttl_minutes: int = 480
    document_storage_root: str = "data/documents"

    # Anthropic is optional at application startup.
    # Provider construction will fail closed if these
    # values are missing when LLM generation is requested.
    anthropic_api_key: SecretStr | None = None
    anthropic_workspace_id: str | None = None
    anthropic_model: str | None = None
    anthropic_max_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
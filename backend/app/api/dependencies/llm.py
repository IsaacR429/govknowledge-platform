from __future__ import annotations

from typing import Annotated

from anthropic import Anthropic
from fastapi import Depends

from app.application.ports.llm_provider import (
    LLMProvider,
)
from app.core.config import (
    Settings,
    get_settings,
)
from app.infrastructure.llm.anthropic_provider import (
    AnthropicLLMProvider,
)


class LLMConfigurationError(
    RuntimeError
):
    pass


def get_llm_provider(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> LLMProvider:
    if settings.anthropic_api_key is None:
        raise LLMConfigurationError(
            "Anthropic API key is not configured."
        )

    if (
        settings.anthropic_model is None
        or not settings.anthropic_model.strip()
    ):
        raise LLMConfigurationError(
            "Anthropic model is not configured."
        )

    if settings.anthropic_max_tokens < 1:
        raise LLMConfigurationError(
            "Anthropic max_tokens must be positive."
        )

    client_kwargs = {
        "api_key": settings.anthropic_api_key.get_secret_value(),
    }

    if (
        settings.anthropic_workspace_id
        and settings.anthropic_workspace_id.strip()
    ):
        client_kwargs["default_headers"] = {
            "anthropic-workspace-id": (
                settings.anthropic_workspace_id.strip()
            )
        }

    client = Anthropic(**client_kwargs)

    return AnthropicLLMProvider(
        client=client,
        model=(
            settings
            .anthropic_model
            .strip()
        ),
        max_tokens=(
            settings.anthropic_max_tokens
        ),
    )

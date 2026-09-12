from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.api.dependencies.llm import (
    LLMConfigurationError,
    get_llm_provider,
)
from app.core.config import Settings
from app.infrastructure.llm.anthropic_provider import (
    AnthropicLLMProvider,
)


def settings(
    *,
    api_key: str | None = "test-key",
    workspace_id: str | None = None,
    model: str | None = "test-model",
    max_tokens: int = 1024,
) -> Settings:
    return Settings(
        anthropic_api_key=(
            SecretStr(api_key)
            if api_key is not None
            else None
        ),
        anthropic_workspace_id=workspace_id,
        anthropic_model=model,
        anthropic_max_tokens=max_tokens,
    )


def test_builds_anthropic_provider():
    provider = get_llm_provider(
        settings()
    )

    assert isinstance(
        provider,
        AnthropicLLMProvider,
    )


def test_missing_api_key_fails_closed():
    with pytest.raises(
        LLMConfigurationError,
        match=(
            "Anthropic API key "
            "is not configured."
        ),
    ):
        get_llm_provider(
            settings(
                api_key=None
            )
        )


def test_missing_model_fails_closed():
    with pytest.raises(
        LLMConfigurationError,
        match=(
            "Anthropic model "
            "is not configured."
        ),
    ):
        get_llm_provider(
            settings(
                model=None
            )
        )


def test_blank_model_fails_closed():
    with pytest.raises(
        LLMConfigurationError,
        match=(
            "Anthropic model "
            "is not configured."
        ),
    ):
        get_llm_provider(
            settings(
                model="   "
            )
        )


def test_invalid_max_tokens_fails_closed():
    with pytest.raises(
        LLMConfigurationError,
        match=(
            "Anthropic max_tokens "
            "must be positive."
        ),
    ):
        get_llm_provider(
            settings(
                max_tokens=0
            )
        )


def test_secret_is_not_exposed_by_settings_repr():
    configured = settings(
        api_key=(
            "super-secret-key-value"
        )
    )

    rendered = repr(configured)

    assert (
        "super-secret-key-value"
        not in rendered
    )

    assert "**********" in rendered


def test_workspace_header_is_passed_when_configured(
    monkeypatch,
):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.api.dependencies.llm.Anthropic",
        FakeAnthropic,
    )

    provider = get_llm_provider(
        settings(
            workspace_id="  wrkspc-test  "
        )
    )

    assert isinstance(
        provider,
        AnthropicLLMProvider,
    )

    assert captured[
        "default_headers"
    ] == {
        "anthropic-workspace-id": "wrkspc-test"
    }


def test_workspace_header_is_omitted_when_not_configured(
    monkeypatch,
):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.api.dependencies.llm.Anthropic",
        FakeAnthropic,
    )

    provider = get_llm_provider(
        settings(
            workspace_id=None
        )
    )

    assert isinstance(
        provider,
        AnthropicLLMProvider,
    )

    assert (
        "default_headers"
        not in captured
    )

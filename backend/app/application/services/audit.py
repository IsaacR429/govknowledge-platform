from __future__ import annotations

from dataclasses import replace
from typing import Mapping
from uuid import UUID

from app.application.ports.audit import (
    AuditEvent,
    AuditRepository,
)


_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "password",
        "raw_password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "bearer",
        "prompt",
        "raw_prompt",
        "query",
        "question",
        "claim",
        "claim_text",
        "claims",
        "answer",
        "response",
        "evidence",
        "evidence_text",
        "chunk_content",
        "content",
        "extracted_text",
        "document_text",
        "source_text",
        "storage_path",
        "file_path",
        "stack_trace",
        "traceback",
    }
)


class UnsafeAuditMetadataError(
    ValueError
):
    pass


class AuditPersistenceError(
    RuntimeError
):
    pass


def _normalized_key(
    key: str,
) -> str:
    return (
        key.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _sanitize_audit_value(
    value: object,
) -> object:
    if isinstance(
        value,
        Mapping,
    ):
        sanitized_mapping: dict[
            str,
            object,
        ] = {}

        for key, nested_value in (
            value.items()
        ):
            normalized = _normalized_key(
                str(key)
            )

            if (
                normalized
                in _FORBIDDEN_METADATA_KEYS
            ):
                raise (
                    UnsafeAuditMetadataError(
                        "Unsafe audit metadata key."
                    )
                )

            sanitized_mapping[
                str(key)
            ] = _sanitize_audit_value(
                nested_value
            )

        return sanitized_mapping

    if isinstance(
        value,
        list,
    ):
        return [
            _sanitize_audit_value(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            _sanitize_audit_value(item)
            for item in value
        )

    return value


def sanitize_audit_metadata(
    metadata: Mapping[
        str,
        object,
    ] | None,
) -> dict[str, object]:
    if metadata is None:
        return {}

    sanitized = _sanitize_audit_value(
        metadata
    )

    if not isinstance(
        sanitized,
        dict,
    ):
        raise UnsafeAuditMetadataError(
            "Audit metadata must be a mapping."
        )

    return sanitized


class AuditService:
    def __init__(
        self,
        repository: AuditRepository,
    ) -> None:
        self._repository = repository

    def record(
        self,
        event: AuditEvent,
    ) -> UUID:
        metadata = (
            sanitize_audit_metadata(
                event.metadata
            )
        )

        safe_event = replace(
            event,
            metadata=metadata,
        )

        try:
            return self._repository.append(
                safe_event
            )
        except Exception as exc:
            raise AuditPersistenceError(
                "Audit event could not be persisted."
            ) from exc

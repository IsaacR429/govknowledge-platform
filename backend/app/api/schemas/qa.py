from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
)


class QuestionAnswerRequest(
    BaseModel
):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    language: str = Field(
        default="fr",
        min_length=2,
        max_length=16,
    )


class QuestionAnswerCitationResponse(
    BaseModel
):
    evidence_id: str

    page_id: UUID
    page_number: int

    document_title: str
    issuing_organization: str
    jurisdiction: str

    cited_text: str


class QuestionAnswerResponse(
    BaseModel
):
    status: Literal[
        "SUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    ]

    answer: str
    correlation_id: UUID

    citations: list[
        QuestionAnswerCitationResponse
    ]

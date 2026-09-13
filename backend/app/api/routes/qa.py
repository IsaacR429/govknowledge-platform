from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.qa import (
    get_citation_presentation_repository,
    get_question_answer_service,
)
from app.api.schemas.qa import (
    QuestionAnswerCitationResponse,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
)
from app.application.ports.citation_presentation_repository import (
    CitationPresentationRepository,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.question_answer import (
    QuestionAnswerService,
)


router = APIRouter(
    tags=["qa"],
)


@router.post(
    "/qa",
    response_model=QuestionAnswerResponse,
)
def answer_question(
    request: QuestionAnswerRequest,
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    service: Annotated[
        QuestionAnswerService,
        Depends(get_question_answer_service),
    ],
    presentation_repository: Annotated[
        CitationPresentationRepository,
        Depends(
            get_citation_presentation_repository
        ),
    ],
) -> QuestionAnswerResponse:
    result = service.answer(
        identity=identity,
        question=request.question,
        language=request.language,
    )

    citation_by_claim_id = {
        item.claim_id: item
        for item in result.citations.citations
    }

    citations: list[
        QuestionAnswerCitationResponse
    ] = []

    for evidence_id in result.cited_evidence_ids:
        claim_citation = (
            citation_by_claim_id.get(
                evidence_id
            )
        )

        if claim_citation is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Validated citation is unavailable."
                ),
            )

        citation = claim_citation.citation

        presentation = (
            presentation_repository.get_by_page_id(
                page_id=citation.page_id,
                document_id=citation.document_id,
            )
        )

        if presentation is None:
            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Citation metadata is unavailable."
                ),
            )

        citations.append(
            QuestionAnswerCitationResponse(
                evidence_id=evidence_id,
                page_id=citation.page_id,
                page_number=(
                    citation.page_number
                ),
                document_title=(
                    presentation.document_title
                ),
                issuing_organization=(
                    presentation.issuing_organization
                ),
                jurisdiction=(
                    presentation.jurisdiction
                ),
                cited_text=(
                    citation.cited_text
                ),
            )
        )

    return QuestionAnswerResponse(
        status=result.status,
        answer=result.answer,
        correlation_id=(
            result.correlation_id
        ),
        citations=citations,
    )

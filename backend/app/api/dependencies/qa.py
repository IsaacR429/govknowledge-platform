from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from anthropic import Anthropic
from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.dependencies.audit import (
    get_audit_service,
)
from app.api.dependencies.llm import (
    get_llm_provider,
)
from app.api.dependencies.source_pages import (
    get_authorization_service,
)
from app.application.ports.citation_presentation_repository import (
    CitationPresentationRepository,
)
from app.application.ports.llm_provider import (
    LLMProvider,
)
from app.application.ports.question_claim_planner import (
    QuestionClaimPlanner,
)
from app.application.ports.table_question_planner import (
    TableQuestionPlanner,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicEvidenceOrchestrationService,
)
from app.application.services.atomic_evidence_policy import (
    AtomicEvidencePolicy,
)
from app.application.services.audit import (
    AuditService,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceService,
)
from app.application.services.citation_mapping import (
    ClaimOriginCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerGenerationService,
)
from app.application.services.table_citation_validation import (
    TableClaimCitationValidator,
)
from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
)
from app.application.services.table_grounded_answer_generation import (
    TableGroundedAnswerGenerationService,
)
from app.application.services.table_question_answer import (
    TableQuestionAnswerService,
)
from app.application.services.table_question_interpretation import (
    ValidatedTableQuestionInterpreter,
)
from app.application.services.table_question_orchestration import (
    TableQuestionOrchestrationService,
)
from app.application.services.question_answer import (
    QuestionAnswerService,
)
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.application.services.validated_citations import (
    ValidatedCitationService,
)
from app.core.config import (
    Settings,
    get_settings,
)
from app.infrastructure.database.repositories.citation_presentation import (
    PostgresCitationPresentationRepository,
)
from app.infrastructure.database.repositories.citation_sources import (
    PostgresCitationSourceRepository,
)
from app.infrastructure.database.repositories.table_evidence import (
    PostgresTableEvidenceRepository,
)
from app.infrastructure.database.session import (
    get_db,
)
from app.infrastructure.llm.anthropic_question_claim_planner import (
    AnthropicQuestionClaimPlanner,
)
from app.infrastructure.llm.anthropic_table_question_planner import (
    AnthropicTableQuestionPlanner,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)
from app.infrastructure.verification.mdeberta_entailment_verifier import (
    MDebertaEntailmentVerifier,
)


NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)

DISCOVERY_TOP_K = 10
EVIDENCE_TOP_K = 5


class QuestionAnswerConfigurationError(
    RuntimeError
):
    pass


@lru_cache(maxsize=1)
def get_qa_embedding_provider(
) -> SentenceTransformerEmbeddingProvider:
    """
    Load the frozen multilingual E5 model once
    per backend process.
    """
    return SentenceTransformerEmbeddingProvider()


@lru_cache(maxsize=1)
def get_qa_entailment_verifier(
) -> MDebertaEntailmentVerifier:
    """
    Load the frozen mDeBERTa NLI model once
    per backend process.
    """
    return MDebertaEntailmentVerifier(
        model_name=NLI_MODEL,
        model_revision=NLI_REVISION,
        local_files_only=True,
    )


def get_question_claim_planner(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> QuestionClaimPlanner:
    if settings.anthropic_api_key is None:
        raise QuestionAnswerConfigurationError(
            "Anthropic API key is not configured."
        )

    if (
        settings.anthropic_model is None
        or not settings.anthropic_model.strip()
    ):
        raise QuestionAnswerConfigurationError(
            "Anthropic model is not configured."
        )

    if settings.anthropic_max_tokens < 1:
        raise QuestionAnswerConfigurationError(
            "Anthropic max_tokens must be positive."
        )

    client_kwargs = {
        "api_key": (
            settings
            .anthropic_api_key
            .get_secret_value()
        ),
    }

    if (
        settings.anthropic_workspace_id
        and settings.anthropic_workspace_id.strip()
    ):
        client_kwargs["default_headers"] = {
            "anthropic-workspace-id": (
                settings
                .anthropic_workspace_id
                .strip()
            )
        }

    client = Anthropic(
        **client_kwargs
    )

    return AnthropicQuestionClaimPlanner(
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


def get_authorized_search_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    authorization: Annotated[
        AuthorizationService,
        Depends(get_authorization_service),
    ],
) -> AuthorizedSearchService:
    return AuthorizedSearchService(
        authorization=authorization,
        embedding_provider=get_qa_embedding_provider(),
        retrieval=PostgresVectorRetrievalAdapter(
            db
        ),
    )



def get_table_question_planner(
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> TableQuestionPlanner:

    if settings.anthropic_api_key is None:
        raise QuestionAnswerConfigurationError(
            "Anthropic API key is not configured."
        )

    if (
        settings.anthropic_model is None
        or not settings.anthropic_model.strip()
    ):
        raise QuestionAnswerConfigurationError(
            "Anthropic model is not configured."
        )

    if settings.anthropic_max_tokens < 1:
        raise QuestionAnswerConfigurationError(
            "Anthropic max_tokens must be positive."
        )

    client_kwargs = {
        "api_key": (
            settings
            .anthropic_api_key
            .get_secret_value()
        ),
    }

    if (
        settings.anthropic_workspace_id
        and settings.anthropic_workspace_id.strip()
    ):
        client_kwargs[
            "default_headers"
        ] = {
            "anthropic-workspace-id": (
                settings
                .anthropic_workspace_id
                .strip()
            )
        }

    client = Anthropic(
        **client_kwargs
    )

    return AnthropicTableQuestionPlanner(
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

def get_question_answer_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    authorization: Annotated[
        AuthorizationService,
        Depends(
            get_authorization_service
        ),
    ],
    audit: Annotated[
        AuditService,
        Depends(get_audit_service),
    ],
    planner: Annotated[
        QuestionClaimPlanner,
        Depends(
            get_question_claim_planner
        ),
    ],
    table_planner: Annotated[
        TableQuestionPlanner,
        Depends(
            get_table_question_planner
        ),
    ],
    llm_provider: Annotated[
        LLMProvider,
        Depends(get_llm_provider),
    ],
) -> QuestionAnswerService:
    entailment_verifier = (
        get_qa_entailment_verifier()
    )

    search = get_authorized_search_service(
        db=db,
        authorization=authorization,
    )

    verification = (
        SentenceEvidenceVerificationService(
            verifier=entailment_verifier
        )
    )

    evidence = (
        AtomicEvidenceOrchestrationService(
            search=search,
            verification=verification,
            policy=AtomicEvidencePolicy(),
            audit=audit,
            top_k=EVIDENCE_TOP_K,
        )
    )

    citation_service = (
        ValidatedCitationService(
            mapper=(
                ClaimOriginCitationMapper()
            ),
            validator=(
                CitationValidator(
                    repository=(
                        PostgresCitationSourceRepository(
                            db
                        )
                    )
                )
            ),
        )
    )

    answer = (
        GroundedAnswerGenerationService(
            citation_service=(
                citation_service
            ),
            provider=llm_provider,
        )
    )

    table_repository = (
        PostgresTableEvidenceRepository(
            db
        )
    )

    table_evidence = (
        AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=(
                table_repository
            ),
        )
    )

    table_citation_validator = (
        TableClaimCitationValidator(
            authorization=authorization,
            repository=(
                table_repository
            ),
        )
    )

    table_answer_generation = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                table_citation_validator
            ),
        )
    )

    table_orchestration = (
        TableQuestionOrchestrationService(
            evidence=table_evidence,
            claim_builder=(
                DirectTableClaimBuilder()
            ),
            answer=(
                table_answer_generation
            ),
        )
    )

    table_interpreter = (
        ValidatedTableQuestionInterpreter(
            planner=table_planner
        )
    )

    table_question_answer = (
        TableQuestionAnswerService(
            evidence=table_evidence,
            interpreter=(
                table_interpreter
            ),
            orchestration=(
                table_orchestration
            ),
        )
    )

    return QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        authorization=authorization,
        audit=audit,
        discovery_top_k=DISCOVERY_TOP_K,
        table_answer=(
            table_question_answer
        ),
    )


def get_citation_presentation_repository(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> CitationPresentationRepository:
    return PostgresCitationPresentationRepository(
        db
    )

from __future__ import annotations

import os
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, text
from sqlalchemy.engine import make_url

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.qa import (
    get_question_answer_service,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.table_question_planner import (
    PlannedTableLookup,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceService,
)
from app.application.services.question_answer import (
    QuestionAnswerService,
)
from app.application.services.table_citation_validation import (
    TableClaimCitationValidator,
)
from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
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
from app.core.config import get_settings
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.models.table_evidence import (
    TableEvidenceRecord,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.repositories.table_evidence import (
    PostgresTableEvidenceRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
    engine,
)
from app.main import create_app


EXPECTED_SYSTEM_ID = (
    "7686228710959620134"
)

SUPPLIER = (
    "ENTREPRISE ZERBO AWA "
    "ET FRERES (EZAF)"
)


# ============================================================
# QA DATABASE GUARD
# ============================================================

def assert_isolated_qa():

    assert (
        os.environ.get(
            "GOVKNOWLEDGE_QA_SYSTEM_ID"
        )
        == EXPECTED_SYSTEM_ID
    )

    expected = make_url(
        os.environ["DATABASE_URL"]
    )

    actual = make_url(
        get_settings().database_url
    )

    assert expected.host == "127.0.0.1"
    assert expected.port == 15432

    assert (
        expected.database
        == "govknowledge_qa"
    )

    assert (
        expected.username
        == "govknowledge_qa"
    )

    assert actual == expected
    assert engine.url == expected

    with engine.connect() as connection:

        database, username, system_id = (
            connection.execute(
                text(
                    "SELECT "
                    "current_database(), "
                    "current_user, "
                    "system_identifier::text "
                    "FROM pg_control_system()"
                )
            )
            .one()
        )

    assert (
        database
        == "govknowledge_qa"
    )

    assert (
        username
        == "govknowledge_qa"
    )

    assert (
        system_id
        == EXPECTED_SYSTEM_ID
    )


# ============================================================
# DETERMINISTIC TEST-ONLY TABLE PLANNER
#
# This replaces ONLY natural-language interpretation.
#
# Everything after its output is production code.
# ============================================================

class DeterministicTablePlanner:

    def plan(
        self,
        context,
    ):

        question = (
            context.question
            .casefold()
        )

        if (
            "09" in question
            or "9 " in question
        ):
            ordinal = 9

        else:
            ordinal = 5

        return PlannedTableLookup(
            section_key="LOT_2",
            ordinal=ordinal,
            column_name=(
                "Soumissionnaires"
            ),
        )


# ============================================================
# TEXT PATH MUST NEVER EXECUTE DURING THESE REQUESTS
# ============================================================

class NeverTextSearch:

    def search(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Text vector retrieval ran "
            "during table HTTP acceptance."
        )


class NeverTextPlanner:

    def plan(
        self,
        request,
    ):
        raise AssertionError(
            "Text claim planner ran "
            "during table HTTP acceptance."
        )


class NeverTextEvidence:

    def evaluate(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Text evidence verification ran "
            "during table HTTP acceptance."
        )


class NeverTextAnswer:

    def generate(
        self,
        **kwargs,
    ):
        raise AssertionError(
            "Text LLM answer generation ran "
            "during table HTTP acceptance."
        )


class RecordingAudit:

    def __init__(
        self,
    ):
        self.events = []

    def record(
        self,
        event,
    ):
        self.events.append(
            event
        )


# ============================================================
# CANONICAL TABLE ATOM
# ============================================================

def build_atom(
    *,
    document_id,
    version_id,
    page_id,
):

    structure_span = (
        TableSourceSpan(
            page_number=4,
            kind=(
                TableSpanKind
                .STRUCTURE
            ),
            source_ref=(
                "http:xref781:"
                "lot2:row05"
            ),
            bbox=None,
            text=None,
        )
    )

    ordinal_cell = (
        TableCellEvidenceAtom(
            column_start=0,
            column_end=1,
            column_name="N°",
            text="05",
            value_source=(
                TableCellValueSource
                .SOURCE_TEXT
            ),
            origin_spans=(
                TableSourceSpan(
                    page_number=4,
                    kind=(
                        TableSpanKind.TEXT
                    ),
                    source_ref=(
                        "http:xref781:"
                        "lot2:row05:ordinal"
                    ),
                    bbox=None,
                    text="05",
                ),
            ),
            structure_spans=(
                structure_span,
            ),
        )
    )

    supplier_cell = (
        TableCellEvidenceAtom(
            column_start=1,
            column_end=2,
            column_name=(
                "Soumissionnaires"
            ),
            text=SUPPLIER,
            value_source=(
                TableCellValueSource
                .SOURCE_TEXT
            ),
            origin_spans=(
                TableSourceSpan(
                    page_number=4,
                    kind=(
                        TableSpanKind.TEXT
                    ),
                    source_ref=(
                        "http:xref781:"
                        "lot2:row05:supplier"
                    ),
                    bbox=None,
                    text=SUPPLIER,
                ),
            ),
            structure_spans=(
                structure_span,
            ),
        )
    )

    return TableEvidenceAtom(
        evidence_id=(
            f"table:{version_id}:"
            "LOT_2/05"
        ),
        document_id=document_id,
        document_version_id=(
            version_id
        ),
        page_id=page_id,
        page_number=4,
        source_table_refs=(
            "xref:781",
        ),
        row_index=10,
        role=TableRowRole.DATA,
        logical_key="LOT_2/05",
        section_key="LOT_2",
        column_names=(
            "N°",
            "Soumissionnaires",
        ),
        cells=(
            ordinal_cell,
            supplier_cell,
        ),
        language="fr",
    )


# ============================================================
# BUILD REAL TABLE APPLICATION STACK FOR ONE REQUEST SESSION
# ============================================================

def build_real_service(
    db,
):

    authorization = (
        AuthorizationService(
            PostgresAuthorizationRepository(
                db
            )
        )
    )

    repository = (
        PostgresTableEvidenceRepository(
            db
        )
    )

    table_evidence = (
        AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=repository,
        )
    )

    citation_validator = (
        TableClaimCitationValidator(
            authorization=authorization,
            repository=repository,
        )
    )

    table_answer = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                citation_validator
            ),
        )
    )

    orchestration = (
        TableQuestionOrchestrationService(
            evidence=table_evidence,
            claim_builder=(
                DirectTableClaimBuilder()
            ),
            answer=table_answer,
        )
    )

    interpreter = (
        ValidatedTableQuestionInterpreter(
            planner=(
                DeterministicTablePlanner()
            )
        )
    )

    table_question_answer = (
        TableQuestionAnswerService(
            evidence=table_evidence,
            interpreter=interpreter,
            orchestration=orchestration,
        )
    )

    return QuestionAnswerService(
        search=NeverTextSearch(),
        planner=NeverTextPlanner(),
        evidence=NeverTextEvidence(),
        answer=NeverTextAnswer(),
        authorization=authorization,
        audit=RecordingAudit(),
        table_answer=(
            table_question_answer
        ),
    )


# ============================================================
# REAL HTTP ACCEPTANCE
# ============================================================

def test_real_http_table_qa():

    assert_isolated_qa()

    setup_db = SessionLocal()

    user_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()
    permission_id = uuid4()

    username = (
        "http-table-qa-"
        + str(user_id)
    )

    identity = CurrentIdentity(
        user_id=user_id,
        username=username,
        display_name=(
            "HTTP Table QA User"
        ),
        email=None,
        department="QA",
        language_preference="fr",
        roles=(
            "EMPLOYEE",
        ),
    )

    try:

        # ====================================================
        # REAL USER
        # ====================================================

        setup_db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "HTTP Table QA User"
                ),
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )
        )

        # ====================================================
        # REAL DOCUMENT / VERSION / PAGE
        # ====================================================

        setup_db.add(
            Document(
                id=document_id,
                title=(
                    "Quotidien Procurement "
                    "HTTP Fixture"
                ),
                description=(
                    "Real HTTP table-QA "
                    "qualification fixture."
                ),
                document_type=(
                    "PROCUREMENT"
                ),
                issuing_organization=(
                    "GovKnowledge QA"
                ),
                jurisdiction=(
                    "Burkina Faso"
                ),
                language="fr",
                publication_date=None,
                source_url=None,
                status="ACTIVE",
                confidentiality_level=(
                    "PUBLIC"
                ),
            )
        )

        setup_db.add(
            DocumentVersion(
                id=version_id,
                document_id=document_id,
                version_number=1,
                original_filename=(
                    "quotidien-http.pdf"
                ),
                storage_path=(
                    "qa-http-table/"
                    "quotidien.pdf"
                ),
                file_hash=(
                    uuid4().hex
                    + uuid4().hex
                ),
                processing_status=(
                    ProcessingStatus
                    .SUCCEEDED
                    .value
                ),
                effective_date=None,
                superseded_at=None,
            )
        )

        setup_db.add(
            DocumentPage(
                id=page_id,
                document_version_id=(
                    version_id
                ),
                page_number=4,
                extracted_text=(
                    "Procurement table "
                    "source page."
                ),
                metadata_json={},
            )
        )

        setup_db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=document_id,
                action=(
                    PermissionAction
                    .READ
                    .value
                ),
            )
        )

        setup_db.commit()

        # ====================================================
        # REAL CANONICAL TABLE PERSISTENCE
        # ====================================================

        atom = build_atom(
            document_id=document_id,
            version_id=version_id,
            page_id=page_id,
        )

        (
            PostgresTableEvidenceRepository(
                setup_db
            )
            .replace_version_atoms(
                version_id=version_id,
                atoms=(
                    atom,
                ),
            )
        )

        print()
        print(
            "REAL POSTGRES TABLE "
            "EVIDENCE PERSISTED: PASS"
        )

        # ====================================================
        # REAL FASTAPI APPLICATION
        # ====================================================

        application = (
            create_app()
        )

        application.dependency_overrides[
            get_current_identity
        ] = lambda: identity

        # Use a fresh request-scoped SQLAlchemy session
        # for the actual QuestionAnswerService.
        def override_service():

            request_db = (
                SessionLocal()
            )

            try:
                yield build_real_service(
                    request_db
                )

            finally:
                request_db.close()

        application.dependency_overrides[
            get_question_answer_service
        ] = override_service

        # ====================================================
        # HTTP TEST 1:
        # REAL FOUND TABLE CELL
        # ====================================================

        with TestClient(
            application,
            raise_server_exceptions=True,
        ) as client:

            response = client.post(
                "/api/v1/qa",
                json={
                    "question": (
                        "Quel soumissionnaire "
                        "correspond au numéro "
                        "05 du lot 2 ?"
                    ),
                    "language": "fr",
                },
            )

            print()
            print(
                "HTTP 05 STATUS:",
                response.status_code,
            )

            print(
                "HTTP 05 BODY:",
                response.json(),
            )

            assert (
                response.status_code
                == 200
            )

            body = response.json()

            assert (
                body["status"]
                == "SUPPORTED"
            )

            assert (
                body["answer"]
                == SUPPLIER
            )

            assert (
                len(
                    body["citations"]
                )
                == 1
            )

            citation = (
                body["citations"][0]
            )

            assert (
                citation[
                    "page_id"
                ]
                == str(
                    page_id
                )
            )

            assert (
                citation[
                    "page_number"
                ]
                == 4
            )

            assert (
                citation[
                    "document_title"
                ]
                == (
                    "Quotidien Procurement "
                    "HTTP Fixture"
                )
            )

            assert (
                citation[
                    "issuing_organization"
                ]
                == "GovKnowledge QA"
            )

            assert (
                citation[
                    "jurisdiction"
                ]
                == "Burkina Faso"
            )

            assert (
                citation[
                    "cited_text"
                ]
                == SUPPLIER
            )

            print(
                "HTTP LOT_2/05 "
                "SUPPORTED: PASS"
            )

            # ================================================
            # HTTP TEST 2:
            # STRUCTURED LOOKUP EXISTS AS INTENT,
            # BUT ROW 09 DOES NOT EXIST.
            #
            # MUST REFUSE.
            # MUST NOT FALL THROUGH TO TEXT RAG.
            # ================================================

            missing = client.post(
                "/api/v1/qa",
                json={
                    "question": (
                        "Quel soumissionnaire "
                        "correspond au numéro "
                        "09 du lot 2 ?"
                    ),
                    "language": "fr",
                },
            )

            print()
            print(
                "HTTP 09 STATUS:",
                missing.status_code,
            )

            print(
                "HTTP 09 BODY:",
                missing.json(),
            )

            assert (
                missing.status_code
                == 200
            )

            missing_body = (
                missing.json()
            )

            assert (
                missing_body["status"]
                == (
                    "INSUFFICIENT_EVIDENCE"
                )
            )

            assert (
                missing_body[
                    "citations"
                ]
                == []
            )

            print(
                "HTTP LOT_2/09 "
                "REFUSAL: PASS"
            )

        print()
        print(
            "=" * 100
        )

        print(
            "REAL HTTP + POSTGRES "
            "TABLE QA ACCEPTANCE: PASS"
        )

        print(
            "=" * 100
        )

    finally:

        setup_db.rollback()

        try:

            setup_db.execute(
                delete(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .document_id
                    == document_id
                )
            )

            setup_db.execute(
                delete(
                    DocumentPermission
                )
                .where(
                    DocumentPermission
                    .user_id
                    == user_id
                )
            )

            setup_db.execute(
                delete(
                    DocumentPage
                )
                .where(
                    DocumentPage.id
                    == page_id
                )
            )

            setup_db.execute(
                delete(
                    DocumentVersion
                )
                .where(
                    DocumentVersion.id
                    == version_id
                )
            )

            setup_db.execute(
                delete(
                    Document
                )
                .where(
                    Document.id
                    == document_id
                )
            )

            setup_db.execute(
                delete(
                    User
                )
                .where(
                    User.id
                    == user_id
                )
            )

            setup_db.commit()

        except Exception:
            setup_db.rollback()
            raise

        finally:
            setup_db.close()

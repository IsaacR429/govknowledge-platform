from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from anthropic import Anthropic
from sqlalchemy import func, select

from app.application.services.question_answer import (
    QuestionAnswerService,
)
from app.infrastructure.llm.anthropic_question_claim_planner import (
    AnthropicQuestionClaimPlanner,
)

# Qualification must use only the already-frozen local model artifacts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.api.dependencies.llm import get_llm_provider
from app.application.ports.identity_provider import CurrentIdentity
from app.application.services.audit import AuditService
from app.application.services.authorization import AuthorizationService
from app.application.services.authorized_search import AuthorizedSearchService
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerGenerationService,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicClaim,
    AtomicEvidenceOrchestrationService,
)
from app.application.services.atomic_evidence_policy import AtomicEvidencePolicy
from app.application.services.citation_mapping import EvidenceSpanCitationMapper
from app.application.services.citation_validation import CitationValidator
from app.application.services.document_indexing import DocumentIndexingService
from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.application.services.validated_citations import (
    InsufficientEvidenceForCitationsError,
    ValidatedCitationService,
)
from app.core.config import get_settings
from app.domain.enums.permission import PermissionAction
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.audit import AuditEventRecord
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import ChunkEmbedding
from app.infrastructure.database.models.identity import User
from app.infrastructure.database.models.permissions import DocumentPermission
from app.infrastructure.database.models.processing import DocumentProcessingJob
from app.infrastructure.database.repositories.audit import PostgresAuditRepository
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.repositories.chunk_embeddings import (
    PostgresChunkEmbeddingRepository,
)
from app.infrastructure.database.repositories.citation_sources import (
    PostgresCitationSourceRepository,
)
from app.infrastructure.database.repositories.document_ingestion import (
    PostgresDocumentIngestionRepository,
)
from app.infrastructure.database.repositories.document_processing import (
    PostgresDocumentProcessingRepository,
)
from app.infrastructure.database.repositories.source_pages import (
    PostgresSourcePageRepository,
)
from app.infrastructure.database.session import SessionLocal
from app.infrastructure.parsing.docling_parser import DoclingDocumentParser
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbeddingProvider,
)
from app.infrastructure.storage.local_file_storage import LocalFileStorage
from app.infrastructure.verification.mdeberta_entailment_verifier import (
    MDebertaEntailmentVerifier,
)


EXPECTED_PAGE_COUNT = 19

EXPECTED_SOURCE_SHA256 = (
    "66e1abd4831ad6018ff8d58c94938e641b61fd7a062bd1c35ccaec6eda2883a5"
)

EXPECTED_DOCLING_BUNDLE_SHA256 = (
    "64253811295ecd8b2b20ed3d3b9f58a0"
    "ebefe90b1882df56bf23d5632b6f6d4d"
)

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)

SUPPORTED_CLAIMS = (
    AtomicClaim(
        claim_id="damaged-baggage-seven-days",
        text=(
            "If checked baggage is damaged, notice must be given "
            "within seven days following receipt of the baggage."
        ),
        language="en",
        required=True,
    ),
    AtomicClaim(
        claim_id="delayed-baggage-twenty-one-days",
        text=(
            "In case of delay, notice must be given within "
            "twenty-one days following delivery of the checked baggage."
        ),
        language="en",
        required=True,
    ),
    AtomicClaim(
        claim_id="lawsuit-two-years",
        text=(
            "A lawsuit arising out of carriage must be filed within "
            "two years, otherwise the right to actions and claims for "
            "loss and damage is extinguished."
        ),
        language="en",
        required=True,
    ),
)

UNSUPPORTED_CLAIM = AtomicClaim(
    claim_id="complimentary-hotel-any-delay",
    text=(
        "Turkish Airlines guarantees every passenger a complimentary "
        "hotel stay whenever any flight is delayed."
    ),
    language="en",
    required=True,
)


def get_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def print_retrieved_chunk(index, chunk) -> None:
    print()
    print(f"  RESULT {index}")
    print("  document_id :", get_attr(chunk, "document_id"))
    print("  chunk_id    :", get_attr(chunk, "chunk_id", "id"))
    print(
        "  page        :",
        get_attr(chunk, "page_number", "page_start"),
    )
    print(
        "  score       :",
        get_attr(
            chunk,
            "score",
            "similarity_score",
            "cosine_similarity",
        ),
    )

    content = get_attr(chunk, "content", "text")
    if content:
        print("  content:")
        print("  " + "-" * 60)
        print(str(content)[:1000])
        print("  " + "-" * 60)


class RecordingQuestionClaimPlanner:
    def __init__(
        self,
        planner,
    ):
        self._planner = planner
        self.calls = []
        self.results = []

    def plan(
        self,
        request,
    ):
        self.calls.append(request)

        result = self._planner.plan(
            request
        )

        self.results.append(result)

        return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "scripts.qualification.live_turkish_airlines_end_to_end "
            "<turkish-airlines-pdf-path>"
        )

    source_pdf = Path(sys.argv[1]).resolve()

    if not source_pdf.is_file():
        raise SystemExit(f"PDF not found: {source_pdf}")

    print("=" * 78)
    print("GOVKNOWLEDGE — TURKISH AIRLINES LIVE END-TO-END QUALIFICATION")
    print("=" * 78)
    print("Document:", source_pdf)

    source_hash = sha256(source_pdf.read_bytes()).hexdigest()

    print()
    print("Source SHA-256:", source_hash)

    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Unexpected source PDF SHA-256: "
            f"{source_hash}"
        )

    artifact_path_raw = os.environ.get(
        "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH"
    )

    if not artifact_path_raw:
        raise RuntimeError(
            "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH is not configured."
        )

    artifact_path = Path(artifact_path_raw).resolve()
    manifest_path = artifact_path / "manifest.json"

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Docling manifest missing: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    bundle_hash = manifest.get("content_sha256")

    if bundle_hash != EXPECTED_DOCLING_BUNDLE_SHA256:
        raise RuntimeError(
            "Unexpected Docling artifact bundle: "
            f"{bundle_hash}"
        )

    print("Docling bundle:", bundle_hash)

    print()
    print("Loading pinned multilingual E5 model...")

    embedding_provider = SentenceTransformerEmbeddingProvider()

    assert embedding_provider.model_name == DEFAULT_MODEL_NAME
    assert embedding_provider.model_revision == DEFAULT_MODEL_REVISION
    assert embedding_provider.dimension == DEFAULT_DIMENSION

    print("Embedding model:", embedding_provider.model_name)
    print("Embedding revision:", embedding_provider.model_revision)
    print("Embedding dimension:", embedding_provider.dimension)

    print()
    print("Loading pinned mDeBERTa NLI model...")

    entailment_verifier = MDebertaEntailmentVerifier(
        model_name=NLI_MODEL,
        model_revision=NLI_REVISION,
        local_files_only=True,
    )

    print("NLI model:", NLI_MODEL)
    print("NLI revision:", NLI_REVISION)

    settings = get_settings()

    storage = LocalFileStorage(
        Path(settings.document_storage_root)
    )

    db = SessionLocal()

    try:
        audit = AuditService(
            repository=PostgresAuditRepository(db)
        )

        ingestion_service = DocumentIngestionService(
            repository=PostgresDocumentIngestionRepository(db),
            storage=storage,
        )

        processing_service = DocumentProcessingService(
            repository=PostgresDocumentProcessingRepository(db),
            storage=storage,
            parser=DoclingDocumentParser(),
            audit=audit,
        )

        indexing_service = DocumentIndexingService(
            repository=PostgresChunkEmbeddingRepository(db),
            embedding_provider=embedding_provider,
        )

        with source_pdf.open("rb") as source:
            created = ingestion_service.create_initial_upload(
                request=CreateDocumentRequest(
                    title=(
                        "Turkish Airlines General Conditions of "
                        "Carriage of Passengers and Baggage"
                    ),
                    description=(
                        "Live end-to-end qualification document "
                        "for GovKnowledge."
                    ),
                    document_type="POLICY",
                    issuing_organization="Turkish Airlines",
                    jurisdiction="International",
                    language="en",
                    publication_date=date(2025, 2, 10),
                    source_url=None,
                    confidentiality_level="PUBLIC",
                    original_filename=source_pdf.name,
                ),
                source=source,
            )

        print()
        print("=== Initial ingestion ===")
        print("Document ID:", created.document_id)
        print("Version ID:", created.version_id)
        print("Processing job ID:", created.processing_job_id)
        print("File SHA-256:", created.file_hash)

        assert created.file_hash == source_hash

        document = db.get(Document, created.document_id)

        assert document is not None
        assert document.status == "ACTIVE"

        print()
        print("Starting production document processing...")

        processed = processing_service.process(
            ProcessDocumentRequest(
                processing_job_id=created.processing_job_id
            )
        )

        db.expire_all()

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert version is not None
        assert job is not None

        assert (
            version.processing_status
            == ProcessingStatus.SUCCEEDED.value
        )

        assert (
            job.status
            == ProcessingStatus.SUCCEEDED.value
        )

        page_count = db.scalar(
            select(func.count(DocumentPage.id)).where(
                DocumentPage.document_version_id
                == created.version_id
            )
        )

        chunk_count = db.scalar(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        assert page_count == EXPECTED_PAGE_COUNT
        assert chunk_count is not None
        assert chunk_count > 0
        assert processed.page_count == page_count
        assert processed.chunk_count == chunk_count

        page_numbers = list(
            db.scalars(
                select(DocumentPage.page_number)
                .where(
                    DocumentPage.document_version_id
                    == created.version_id
                )
                .order_by(DocumentPage.page_number)
            )
        )

        assert page_numbers == list(
            range(1, EXPECTED_PAGE_COUNT + 1)
        )

        chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_version_id
                    == created.version_id
                )
                .order_by(DocumentChunk.chunk_index)
            )
        )

        pages_by_id = {
            page.id: page
            for page in db.scalars(
                select(DocumentPage).where(
                    DocumentPage.document_version_id
                    == created.version_id
                )
            )
        }

        assert len(chunks) == chunk_count

        for expected_index, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected_index

            page = pages_by_id.get(chunk.page_id)

            assert page is not None
            assert chunk.page_start == page.page_number
            assert chunk.page_end == page.page_number

            assert (
                chunk.metadata_json["page_number"]
                == page.page_number
            )

        print()
        print("=== Processing result ===")
        print("Pages:", page_count)
        print("Chunks:", chunk_count)
        print("Version status:", version.processing_status)
        print("Job status:", job.status)
        print("Page lineage: PASS")

        print()
        print("Starting production embedding indexing...")

        indexed = indexing_service.index_version(
            version_id=created.version_id
        )

        db.expire_all()

        embedding_count = db.scalar(
            select(func.count(ChunkEmbedding.id))
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        assert embedding_count is not None
        assert indexed.indexed_count == chunk_count
        assert embedding_count == chunk_count
        assert indexed.model_name == DEFAULT_MODEL_NAME
        assert indexed.model_revision == DEFAULT_MODEL_REVISION

        print()
        print("=== Embedding result ===")
        print("Indexed chunks:", indexed.indexed_count)
        print("Embedding rows:", embedding_count)
        print("Model:", indexed.model_name)
        print("Revision:", indexed.model_revision)

        username = (
            "govknowledge-turkish-airlines-"
            f"{created.document_id.hex[:8]}"
        )

        user = User(
            username=username,
            display_name="Turkish Airlines Qualification User",
            email=None,
            department="Qualification",
            language_preference="en",
            is_active=True,
        )

        db.add(user)
        db.flush()

        permission = DocumentPermission(
            user_id=user.id,
            document_id=created.document_id,
            action=PermissionAction.READ.value,
        )

        db.add(permission)
        db.commit()

        identity = CurrentIdentity(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            email=user.email,
            department=user.department,
            language_preference=user.language_preference,
            roles=(),
        )

        authorization = AuthorizationService(
            PostgresAuthorizationRepository(db)
        )

        scope = authorization.retrieval_scope(identity)

        print()
        print("=== Authorization scope ===")
        print("Allowed document IDs:", scope.document_ids)

        assert scope.document_ids == frozenset(
            {created.document_id}
        )

        print("Authorization scope: PASS (new document only)")

        search = AuthorizedSearchService(
            authorization=authorization,
            embedding_provider=embedding_provider,
            retrieval=PostgresVectorRetrievalAdapter(db),
        )

        verification = SentenceEvidenceVerificationService(
            verifier=entailment_verifier
        )

        orchestration = AtomicEvidenceOrchestrationService(
            search=search,
            verification=verification,
            policy=AtomicEvidencePolicy(),
            audit=audit,
            top_k=5,
        )

        print()
        print("=" * 78)
        print("SUPPORTED CLAIMS — REAL EVIDENCE PIPELINE")
        print("=" * 78)

        supported_result = orchestration.evaluate(
            identity=identity,
            claims=SUPPORTED_CLAIMS,
        )

        assert supported_result.is_sufficient

        for claim_result in supported_result.claim_results:
            print()
            print("CLAIM:", claim_result.claim.claim_id)
            print(claim_result.claim.text)

            retrieved = claim_result.retrieved_evidence

            assert retrieved

            pages = []

            for index, chunk in enumerate(
                retrieved,
                start=1,
            ):
                print_retrieved_chunk(
                    index,
                    chunk,
                )

                assert (
                    get_attr(chunk, "document_id")
                    == created.document_id
                )

                page_number = get_attr(
                    chunk,
                    "page_number",
                    "page_start",
                )

                if page_number is not None:
                    pages.append(page_number)

            print("Retrieved pages:", pages)

            assert 19 in pages

        print()
        print("Supported evidence decision: SUFFICIENT")

        citation_service = ValidatedCitationService(
            mapper=EvidenceSpanCitationMapper(),
            validator=CitationValidator(
                repository=PostgresCitationSourceRepository(db)
            ),
        )

        validated = citation_service.build(
            supported_result.assessment
        )

        print()
        print("=== Validated citations ===")

        assert len(validated.citations) == len(
            SUPPORTED_CLAIMS
        )

        for item in validated.citations:
            citation = item.citation

            print()
            print("Claim ID:", item.claim_id)
            print("Page:", citation.page_number)
            print("Chunk ID:", citation.chunk_id)
            print("Text:", citation.cited_text)

            assert (
                citation.document_id
                == created.document_id
            )

            assert (
                citation.document_version_id
                == created.version_id
            )

            assert citation.page_number == 19
            assert citation.cited_text.strip()
            assert citation.end_char > citation.start_char

        print()
        print("Canonical citation validation: PASS")

        print()
        print("=" * 78)
        print("REAL GROUNDED LLM ANSWER")
        print("=" * 78)

        answer_service = GroundedAnswerGenerationService(
            citation_service=citation_service,
            provider=get_llm_provider(get_settings()),
        )

        print()
        print("=" * 78)
        print(
            "NATURAL QUESTION — FULL PLANNER "
            "TO GROUNDED ANSWER PIPELINE"
        )
        print("=" * 78)

        settings = get_settings()

        if settings.anthropic_api_key is None:
            raise RuntimeError(
                "Anthropic API key is not configured."
            )

        if (
            settings.anthropic_model is None
            or not settings.anthropic_model.strip()
        ):
            raise RuntimeError(
                "Anthropic model is not configured."
            )

        planner_client_kwargs = {
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
            planner_client_kwargs[
                "default_headers"
            ] = {
                "anthropic-workspace-id": (
                    settings
                    .anthropic_workspace_id
                    .strip()
                )
            }

        planner = AnthropicQuestionClaimPlanner(
            client=Anthropic(
                **planner_client_kwargs
            ),
            model=(
                settings
                .anthropic_model
                .strip()
            ),
            max_tokens=(
                settings.anthropic_max_tokens
            ),
        )

        planner = RecordingQuestionClaimPlanner(
            planner
        )

        question_answer_service = (
            QuestionAnswerService(
                search=search,
                planner=planner,
                evidence=orchestration,
                answer=answer_service,
                discovery_top_k=10,
            )
        )

        natural_question = (
            "Quel est le delai pour signaler "
            "un bagage enregistre endommage ?"
        )

        natural_answer = (
            question_answer_service.answer(
                identity=identity,
                question=natural_question,
                language="fr",
            )
        )

        print()
        print("=== NATURAL QUESTION DIAGNOSTIC ===")

        print(
            "Planner calls:",
            len(planner.calls),
        )

        if planner.calls:
            planner_request = (
                planner.calls[-1]
            )

            print(
                "Planner source language:",
                planner_request.source_language,
            )

            print(
                "Planner evidence count:",
                len(
                    planner_request.evidence
                ),
            )

            for index, item in enumerate(
                planner_request.evidence,
                start=1,
            ):
                print()
                print(
                    f"DISCOVERY EVIDENCE {index}"
                )
                print(
                    "language:",
                    item.language,
                )
                print(
                    "content:",
                )
                print(
                    item.content
                )

        if planner.results:
            print()
            print("PLANNED CLAIMS:")

            for index, claim in enumerate(
                planner.results[-1],
                start=1,
            ):
                print(
                    f"{index}. {claim.text}"
                )

            if not planner.results[-1]:
                print("<NONE>")

        print()
        print(
            "Natural-question status:",
            natural_answer.status,
        )

        print()
        print("ANSWER:")
        print(
            natural_answer.answer
            .encode("unicode_escape")
            .decode()
        )

        print()
        print("CITED EVIDENCE IDS:")
        print(
            natural_answer.cited_evidence_ids
        )

        print()
        print(
            "Correlation ID:",
            natural_answer.correlation_id,
        )

        assert (
            natural_answer.status
            == "SUPPORTED"
        )

        assert (
            natural_answer.citations.citations
        )

        assert (
            natural_answer.cited_evidence_ids
        )

        print()
        print(
            "Natural-language question pipeline: PASS"
        )

        answer_result = answer_service.generate(
            question=(
                "Quel est le delai pour signaler "
                "un bagage enregistre endommage ?"
            ),
            evidence_result=supported_result,
            language="fr",
        )

        assert answer_result.status == "SUPPORTED"
        assert answer_result.citations.citations
        assert answer_result.cited_evidence_ids

        print("Grounded answer status:", answer_result.status)

        print()
        print("ANSWER:")
        print(
            answer_result.answer
            .encode("unicode_escape")
            .decode("ascii")
        )

        print()
        print("CITED EVIDENCE IDS:")
        print(answer_result.cited_evidence_ids)

        print()
        print("Grounded Claude answer: PASS")

        source_page_service = AuthorizedSourcePageService(
            repository=PostgresSourcePageRepository(db),
            authorization=authorization,
            audit=audit,
        )

        source_pages = {}

        for item in validated.citations:
            page_id = item.citation.page_id

            if page_id in source_pages:
                continue

            page = source_page_service.get(
                identity=identity,
                page_id=page_id,
                correlation_id=(
                    supported_result.correlation_id
                ),
            )

            source_pages[page_id] = page

            assert (
                page.document_id
                == created.document_id
            )

            assert (
                page.document_version_id
                == created.version_id
            )

            assert page.page_number == 19

            assert page.document_title == (
                "Turkish Airlines General Conditions of "
                "Carriage of Passengers and Baggage"
            )

            assert (
                page.issuing_organization
                == "Turkish Airlines"
            )

            assert (
                page.jurisdiction
                == "International"
            )

            assert page.document_language == "en"

            assert not hasattr(
                page,
                "storage_path",
            )

            print()
            print("=== Authorized source page ===")
            print("Page ID:", page.page_id)
            print("Page number:", page.page_number)
            print("Document:", page.document_title)
            print("Issuer:", page.issuing_organization)
            print("Jurisdiction:", page.jurisdiction)

            print(
                "Extracted text preview:",
                page.extracted_text[:700],
            )

        print()
        print("Authorized source-page retrieval: PASS")

        print()
        print("=" * 78)
        print("UNSUPPORTED CLAIM — REFUSAL CONTROL")
        print("=" * 78)
        print(UNSUPPORTED_CLAIM.text)

        unsupported_result = orchestration.evaluate(
            identity=identity,
            claims=(UNSUPPORTED_CLAIM,),
        )

        print(
            "Unsupported evidence decision:",
            (
                "SUFFICIENT"
                if unsupported_result.is_sufficient
                else "INSUFFICIENT"
            ),
        )

        assert not unsupported_result.is_sufficient

        try:
            citation_service.build(
                unsupported_result.assessment
            )

        except InsufficientEvidenceForCitationsError:
            print(
                "Citation generation blocked for "
                "INSUFFICIENT evidence: PASS"
            )

        else:
            raise AssertionError(
                "INSUFFICIENT evidence unexpectedly "
                "produced citations."
            )

        db.expire_all()

        audit_rows = list(
            db.scalars(
                select(AuditEventRecord)
                .where(
                    AuditEventRecord.actor_user_id
                    == user.id
                )
                .order_by(
                    AuditEventRecord.occurred_at,
                    AuditEventRecord.id,
                )
            )
        )

        print()
        print("=== Persisted audit records ===")
        print("Audit row count:", len(audit_rows))

        for row in audit_rows:
            print(
                " -",
                row.event_type,
                "|",
                row.outcome,
                "| document:",
                row.document_id,
                "| correlation:",
                row.correlation_id,
                "| error:",
                row.error_code,
            )

        assert len(audit_rows) >= 3

        assert any(
            row.correlation_id
            == supported_result.correlation_id
            for row in audit_rows
        )

        assert any(
            row.correlation_id
            == unsupported_result.correlation_id
            for row in audit_rows
        )

        assert any(
            row.document_id
            == created.document_id
            and row.resource_type
            == "source_page"
            and row.correlation_id
            == supported_result.correlation_id
            for row in audit_rows
        )

        serialized_metadata = "\n".join(
            json.dumps(
                row.metadata_json or {},
                sort_keys=True,
                default=str,
            )
            for row in audit_rows
        )

        for claim in (
            *SUPPORTED_CLAIMS,
            UNSUPPORTED_CLAIM,
        ):
            assert (
                claim.text
                not in serialized_metadata
            )

        assert (
            str(source_pdf)
            not in serialized_metadata
        )

        print("Audit persistence/sanitization: PASS")

        print()
        print("=" * 78)
        print(
            "TURKISH AIRLINES LIVE END-TO-END "
            "QUALIFICATION: PASS"
        )
        print("=" * 78)

        print("Document ID:", created.document_id)
        print("Version ID:", created.version_id)
        print(
            "Processing job ID:",
            created.processing_job_id,
        )
        print("Pages:", page_count)
        print("Chunks:", chunk_count)
        print("Embeddings:", embedding_count)
        print(
            "Validated citations:",
            len(validated.citations),
        )
        print(
            "Source pages checked:",
            len(source_pages),
        )
        print("Audit rows:", len(audit_rows))

        print(
            "Records intentionally retained for final "
            "qualification traceability."
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()

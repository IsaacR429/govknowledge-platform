from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.application.ports.identity_provider import CurrentIdentity
from app.application.services.authorization import AuthorizationService
from app.domain.enums.permission import PermissionAction
from app.domain.enums.role import RoleName
from app.infrastructure.database.models.documents import Document
from app.infrastructure.database.models.identity import Role, User
from app.infrastructure.database.models.permissions import DocumentPermission
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.session import SessionLocal


EMPLOYEE_USERNAME = "employee.demo"

ALLOWED_TITLE = "Guide des procédures administratives"
DENIED_TITLE = "Circulaire budgétaire interne"


def ensure_document(
    db,
    *,
    title: str,
    issuing_organization: str,
    jurisdiction: str,
) -> Document:
    document = db.scalar(
        select(Document).where(
            Document.title == title
        )
    )

    if document is None:
        document = Document(
            title=title,
            description="Document de démonstration GovKnowledge.",
            document_type="POLICY",
            issuing_organization=issuing_organization,
            jurisdiction=jurisdiction,
            language="fr",
            status="ACTIVE",
            confidentiality_level="INTERNAL",
        )

        db.add(document)
        db.flush()

    return document


def main() -> None:
    with SessionLocal() as db:
        employee_role = db.scalar(
            select(Role).where(
                Role.name == RoleName.EMPLOYEE.value
            )
        )

        if employee_role is None:
            raise SystemExit(
                "EMPLOYEE role is missing. "
                "Run seed_demo_identity first."
            )

        employee = db.scalar(
            select(User)
            .options(selectinload(User.roles))
            .where(
                User.username == EMPLOYEE_USERNAME
            )
        )

        if employee is None:
            employee = User(
                username=EMPLOYEE_USERNAME,
                display_name="Demo Employee",
                email=None,
                department="Administration",
                language_preference="fr",
                is_active=True,
            )

            employee.roles.append(employee_role)

            db.add(employee)
            db.flush()

        elif all(
            role.name != RoleName.EMPLOYEE.value
            for role in employee.roles
        ):
            employee.roles.append(employee_role)

        allowed_document = ensure_document(
            db,
            title=ALLOWED_TITLE,
            issuing_organization=(
                "Ministère de la Transition Digitale"
            ),
            jurisdiction="Burkina Faso",
        )

        denied_document = ensure_document(
            db,
            title=DENIED_TITLE,
            issuing_organization=(
                "Ministère de l'Économie et des Finances"
            ),
            jurisdiction="Burkina Faso",
        )

        permission = db.scalar(
            select(DocumentPermission).where(
                DocumentPermission.user_id
                == employee.id,
                DocumentPermission.document_id
                == allowed_document.id,
                DocumentPermission.action
                == PermissionAction.READ.value,
            )
        )

        if permission is None:
            db.add(
                DocumentPermission(
                    user_id=employee.id,
                    document_id=allowed_document.id,
                    action=PermissionAction.READ.value,
                )
            )

        db.commit()

        identity = CurrentIdentity(
            user_id=employee.id,
            username=employee.username,
            display_name=employee.display_name,
            email=employee.email,
            department=employee.department,
            language_preference=employee.language_preference,
            roles=tuple(
                sorted(
                    role.name
                    for role in employee.roles
                )
            ),
        )

        repository = PostgresAuthorizationRepository(db)
        service = AuthorizationService(repository)

        scope = service.retrieval_scope(identity)

        print(
            "Employee:",
            employee.username,
        )
        print(
            "Allowed document:",
            allowed_document.id,
            allowed_document.title,
        )
        print(
            "Denied document:",
            denied_document.id,
            denied_document.title,
        )
        print(
            "Authorization scope:",
            sorted(
                str(document_id)
                for document_id
                in scope.document_ids
            ),
        )

        assert scope.allows(
            allowed_document.id
        )

        assert not scope.allows(
            denied_document.id
        )

        assert scope.document_ids == frozenset(
            {
                allowed_document.id,
            }
        )

        print(
            "PostgreSQL authorization proof: PASS"
        )


if __name__ == "__main__":
    main()

from getpass import getpass

from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums.role import RoleName
from app.infrastructure.database.models.authentication import (
    LocalCredential,
)
from app.infrastructure.database.models.identity import (
    Role,
    User,
)
from app.infrastructure.database.session import SessionLocal


password_hasher = PasswordHash.recommended()


ROLE_DESCRIPTIONS = {
    RoleName.ADMIN: "Manages users, documents, and access.",
    RoleName.EMPLOYEE: "Searches and asks questions over authorized documents.",
    RoleName.AUDITOR: "Reviews evidence, citations, and audit activity.",
}


def main() -> None:
    with SessionLocal() as db:
        roles: dict[RoleName, Role] = {}

        for role_name, description in ROLE_DESCRIPTIONS.items():
            role = db.scalar(
                select(Role).where(
                    Role.name == role_name.value
                )
            )

            if role is None:
                role = Role(
                    name=role_name.value,
                    description=description,
                )
                db.add(role)
                db.flush()

            roles[role_name] = role

        admin = db.scalar(
            select(User)
            .options(selectinload(User.roles))
            .where(User.username == "admin")
        )

        user_created = False

        if admin is None:
            admin = User(
                username="admin",
                display_name="GovKnowledge Administrator",
                email=None,
                department="Administration",
                language_preference="fr",
                is_active=True,
            )

            admin.roles.append(
                roles[RoleName.ADMIN]
            )

            db.add(admin)
            db.flush()

            user_created = True

        elif all(
            role.name != RoleName.ADMIN.value
            for role in admin.roles
        ):
            admin.roles.append(
                roles[RoleName.ADMIN]
            )

        credential = db.get(
            LocalCredential,
            admin.id,
        )

        credential_created = False

        if credential is None:
            password = getpass(
                "Enter the demo administrator password: "
            )

            if len(password) < 12:
                raise SystemExit(
                    "Password must contain at least "
                    "12 characters."
                )

            password_confirmation = getpass(
                "Confirm the password: "
            )

            if password != password_confirmation:
                raise SystemExit(
                    "Passwords do not match."
                )

            credential = LocalCredential(
                user_id=admin.id,
                password_hash=password_hasher.hash(
                    password
                ),
            )

            db.add(credential)
            credential_created = True

        db.commit()

        print(
            "Roles ensured: "
            "ADMIN, EMPLOYEE, AUDITOR"
        )

        if user_created:
            print("Demo administrator created: admin")
        else:
            print("Demo administrator already exists: admin")

        if credential_created:
            print("Local administrator credential created")
        else:
            print(
                "Existing administrator credential preserved"
            )


if __name__ == "__main__":
    main()

from getpass import getpass

from pwdlib import PasswordHash
from sqlalchemy import select

from app.domain.enums.role import RoleName
from app.infrastructure.database.models.authentication import LocalCredential
from app.infrastructure.database.models.identity import Role, User
from app.infrastructure.database.session import SessionLocal


USERNAME = "employee.isolation.p0"


def main() -> None:
    with SessionLocal() as db:
        existing = db.scalar(
            select(User.id).where(User.username == USERNAME)
        )
        if existing is not None:
            raise SystemExit(
                "Test account already exists; no changes made."
            )

        role = db.scalar(
            select(Role).where(Role.name == RoleName.EMPLOYEE.value)
        )
        if role is None:
            raise SystemExit("EMPLOYEE role not found; no changes made.")

        password = getpass("Choose a test employee password: ")
        if len(password) < 12:
            raise SystemExit("Use at least 12 characters; no changes made.")

        confirmation = getpass("Confirm password: ")
        if password != confirmation:
            raise SystemExit("Passwords differ; no changes made.")

        user = User(
            username=USERNAME,
            display_name="Isolation Test Employee",
            email=None,
            department="Testing",
            language_preference="fr",
            is_active=True,
            roles=[role],
        )
        db.add(user)
        db.flush()

        db.add(
            LocalCredential(
                user_id=user.id,
                password_hash=PasswordHash.recommended().hash(password),
            )
        )
        db.commit()

        print("Created test employee:", USERNAME)
        print("Test employee ID:", user.id)


if __name__ == "__main__":
    main()

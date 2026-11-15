"""Provision the seven GovKnowledge QA identities.

Run only against a dedicated, isolated QA database.
Existing accounts and credentials are never overwritten.
"""

import argparse
import re
from getpass import getpass

from pwdlib import PasswordHash
from sqlalchemy import select, text
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


PASSWORD_HASHER = PasswordHash.recommended()

ROLE_DESCRIPTIONS = {
    RoleName.ADMIN: "Manages users, documents, and access.",
    RoleName.EMPLOYEE: (
        "Searches and asks questions over authorized documents."
    ),
    RoleName.AUDITOR: (
        "Reviews evidence, citations, and audit activity."
    ),
}

ACCOUNTS = (
    ("qa.admin.01", "QA Administrator 1", RoleName.ADMIN),
    ("qa.admin.02", "QA Administrator 2", RoleName.ADMIN),
    ("qa.employee.01", "QA Employee 1", RoleName.EMPLOYEE),
    ("qa.employee.02", "QA Employee 2", RoleName.EMPLOYEE),
    ("qa.employee.03", "QA Employee 3", RoleName.EMPLOYEE),
    ("qa.employee.04", "QA Employee 4", RoleName.EMPLOYEE),
    ("qa.auditor.01", "QA Auditor 1", RoleName.AUDITOR),
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expected-database",
        required=True,
    )
    parser.add_argument(
        "--expected-host",
        required=True,
    )
    parser.add_argument(
        "--expected-user",
        required=True,
    )
    parser.add_argument(
        "--expected-port",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    with SessionLocal() as db:

        # --------------------------------------------------
        # 1. Validate the actual database connection.
        # --------------------------------------------------

        engine = db.get_bind()

        if engine.dialect.name != "postgresql":
            raise SystemExit(
                "Refusing to run: PostgreSQL required."
            )

        actual_host = engine.url.host
        actual_port = engine.url.port or 5432

        actual_database, actual_user = db.execute(
            text(
                "SELECT current_database(), current_user"
            )
        ).one()

        if (
            actual_database != args.expected_database
            or actual_host != args.expected_host
            or actual_user != args.expected_user
            or actual_port != args.expected_port
        ):
            raise SystemExit(
                "Database identity mismatch. "
                "No changes made."
            )

        # Require an explicit QA/test database name.
        if not re.search(
            r"(?:^|[_-])(qa|test|testing|isolation)(?:[_-]|$)",
            actual_database,
            re.IGNORECASE,
        ):
            raise SystemExit(
                "Database name does not identify a QA "
                "or test environment. No changes made."
            )

        print("\nVerified database connection:")
        print("Host:", actual_host)
        print("Database:", actual_database)
        print("Database user:", actual_user)

        confirmation = input(
            f"\nType PROVISION {actual_database} "
            "to continue: "
        )

        if confirmation.strip() != (
            f"PROVISION {actual_database}"
        ):
            raise SystemExit(
                "Provisioning cancelled. No changes made."
            )

        # --------------------------------------------------
        # 2. Inspect existing QA accounts.
        # --------------------------------------------------

        usernames = [
            account[0]
            for account in ACCOUNTS
        ]

        existing_users = {
            user.username: user
            for user in db.scalars(
                select(User)
                .options(selectinload(User.roles))
                .where(User.username.in_(usernames))
            ).all()
        }

        # Fail rather than silently modifying an account
        # that does not match the expected QA identity.

        for username, display_name, role_name in ACCOUNTS:
            existing = existing_users.get(username)

            if existing is None:
                continue

            existing_roles = {
                role.name
                for role in existing.roles
            }

            if (
                existing_roles != {role_name.value}
                or existing.department != "QA"
                or existing.display_name != display_name
                or not existing.is_active
            ):
                raise SystemExit(
                    f"Existing account {username} does "
                    "not match its expected QA profile. "
                    "No changes made."
                )

            credential = db.get(
                LocalCredential,
                existing.id,
            )

            if credential is None:
                raise SystemExit(
                    f"Existing account {username} has "
                    "no local credential. "
                    "No changes made."
                )

        # --------------------------------------------------
        # 3. Collect credentials for missing accounts.
        # --------------------------------------------------

        new_password_hashes = {}

        for username, _, _ in ACCOUNTS:

            if username in existing_users:
                print(
                    f"Preserving existing account: "
                    f"{username}"
                )
                continue

            print(f"\nCreating credential for {username}")

            password = getpass(
                "Password (minimum 12 characters): "
            )

            if len(password) < 12:
                raise SystemExit(
                    "Password too short. "
                    "No changes made."
                )

            confirmation = getpass(
                "Confirm password: "
            )

            if password != confirmation:
                raise SystemExit(
                    "Passwords do not match. "
                    "No changes made."
                )

            new_password_hashes[username] = (
                PASSWORD_HASHER.hash(password)
            )

            del password
            del confirmation

        # --------------------------------------------------
        # 4. Ensure all three roles exist.
        # --------------------------------------------------

        roles = {}

        for role_name, description in (
            ROLE_DESCRIPTIONS.items()
        ):
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

        # --------------------------------------------------
        # 5. Create only missing QA accounts.
        # --------------------------------------------------

        created = []

        for username, display_name, role_name in ACCOUNTS:

            if username in existing_users:
                continue

            user = User(
                username=username,
                display_name=display_name,
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )

            user.roles.append(
                roles[role_name]
            )

            db.add(user)
            db.flush()

            credential = LocalCredential(
                user_id=user.id,
                password_hash=(
                    new_password_hashes[username]
                ),
            )

            db.add(credential)
            created.append(username)

        # --------------------------------------------------
        # 6. Commit the provisioning transaction.
        # --------------------------------------------------

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        print("\nProvisioning completed.")
        print("Accounts created:", len(created))
        print(
            "Existing accounts preserved:",
            len(existing_users),
        )

        for username in created:
            print("Created:", username)


if __name__ == "__main__":
    main()

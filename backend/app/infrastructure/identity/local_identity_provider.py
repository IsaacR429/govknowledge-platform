from datetime import datetime, timedelta, timezone
from hashlib import sha256
import secrets

from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.application.ports.identity_provider import (
    CurrentIdentity,
    IdentityProvider,
    LoginResult,
)
from app.core.config import get_settings
from app.domain.exceptions.authentication import (
    InvalidCredentialsError,
    InvalidSessionError,
)
from app.infrastructure.database.models.authentication import (
    AuthSession,
    LocalCredential,
)
from app.infrastructure.database.models.identity import User


_password_hasher = PasswordHash.recommended()

# Used so an unknown username still performs an Argon2 verification.
# This reduces obvious username-enumeration timing differences.
_dummy_password_hash = _password_hasher.hash(
    "GovKnowledge-Dummy-Credential"
)


class LocalIdentityProvider(IdentityProvider):
    """PostgreSQL-backed identity provider for the P0 demo."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._settings = get_settings()

    @staticmethod
    def _hash_token(access_token: str) -> str:
        return sha256(
            access_token.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _to_identity(user: User) -> CurrentIdentity:
        return CurrentIdentity(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            email=user.email,
            department=user.department,
            language_preference=user.language_preference,
            roles=tuple(
                sorted(role.name for role in user.roles)
            ),
        )

    def _get_user(
        self,
        username: str,
    ) -> User | None:
        statement = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.username == username)
        )

        return self._db.scalar(statement)

    def login(
        self,
        username: str,
        password: str,
    ) -> LoginResult:
        user = self._get_user(username)

        credential = (
            self._db.get(LocalCredential, user.id)
            if user is not None
            else None
        )

        password_hash = (
            credential.password_hash
            if credential is not None
            else _dummy_password_hash
        )

        valid_password = _password_hasher.verify(
            password,
            password_hash,
        )

        if (
            user is None
            or credential is None
            or not valid_password
            or not user.is_active
        ):
            raise InvalidCredentialsError(
                "Invalid username or password."
            )

        access_token = secrets.token_urlsafe(32)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(
            minutes=self._settings.auth_session_ttl_minutes
        )

        session = AuthSession(
            user_id=user.id,
            token_hash=self._hash_token(access_token),
            expires_at=expires_at,
        )

        self._db.add(session)
        self._db.commit()

        return LoginResult(
            access_token=access_token,
            expires_at=expires_at,
            identity=self._to_identity(user),
        )

    def resolve_token(
        self,
        access_token: str,
    ) -> CurrentIdentity:
        token_hash = self._hash_token(access_token)

        statement = select(AuthSession).where(
            AuthSession.token_hash == token_hash
        )

        auth_session = self._db.scalar(statement)

        now = datetime.now(timezone.utc)

        if (
            auth_session is None
            or auth_session.revoked_at is not None
            or auth_session.expires_at <= now
        ):
            raise InvalidSessionError(
                "Authentication session is invalid."
            )

        statement = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == auth_session.user_id)
        )

        user = self._db.scalar(statement)

        if user is None or not user.is_active:
            raise InvalidSessionError(
                "Authentication session is invalid."
            )

        return self._to_identity(user)

    def logout(
        self,
        access_token: str,
    ) -> None:
        token_hash = self._hash_token(access_token)

        statement = select(AuthSession).where(
            AuthSession.token_hash == token_hash
        )

        auth_session = self._db.scalar(statement)

        # Logout is deliberately idempotent.
        if auth_session is None:
            return

        if auth_session.revoked_at is None:
            auth_session.revoked_at = datetime.now(
                timezone.utc
            )
            self._db.commit()

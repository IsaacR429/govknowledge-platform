from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class CurrentIdentity:
    user_id: UUID
    username: str
    display_name: str
    email: str | None
    department: str | None
    language_preference: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class LoginResult:
    access_token: str
    expires_at: datetime
    identity: CurrentIdentity


class IdentityProvider(Protocol):
    def login(
        self,
        username: str,
        password: str,
    ) -> LoginResult:
        ...

    def resolve_token(
        self,
        access_token: str,
    ) -> CurrentIdentity:
        ...

    def logout(
        self,
        access_token: str,
    ) -> None:
        ...

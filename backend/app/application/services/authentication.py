from app.application.ports.identity_provider import (
    CurrentIdentity,
    IdentityProvider,
    LoginResult,
)


class AuthenticationService:
    def __init__(
        self,
        identity_provider: IdentityProvider,
    ) -> None:
        self._identity_provider = identity_provider

    def login(
        self,
        username: str,
        password: str,
    ) -> LoginResult:
        return self._identity_provider.login(
            username=username,
            password=password,
        )

    def current_identity(
        self,
        access_token: str,
    ) -> CurrentIdentity:
        return self._identity_provider.resolve_token(
            access_token
        )

    def logout(
        self,
        access_token: str,
    ) -> None:
        self._identity_provider.logout(access_token)

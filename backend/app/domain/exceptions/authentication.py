class AuthenticationError(Exception):
    """Authentication failed without exposing sensitive detail."""


class InvalidCredentialsError(AuthenticationError):
    """Username/password authentication failed."""


class InvalidSessionError(AuthenticationError):
    """Bearer session is invalid, expired, or revoked."""

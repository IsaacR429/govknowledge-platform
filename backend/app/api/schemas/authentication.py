from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class IdentityResponse(BaseModel):
    user_id: UUID
    username: str
    display_name: str
    email: str | None
    department: str | None
    language_preference: str
    roles: list[str]


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: IdentityResponse

from uuid import UUID

from pydantic import BaseModel


class DocumentReadPermissionUserResponse(BaseModel):
    user_id: UUID
    username: str
    display_name: str
    department: str | None
    is_active: bool
    roles: list[str]
    has_read: bool


class DocumentReadPermissionListResponse(BaseModel):
    document_id: UUID
    items: list[DocumentReadPermissionUserResponse]

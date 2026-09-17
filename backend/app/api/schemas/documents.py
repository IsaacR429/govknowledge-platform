from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID
    file_hash: str
    size_bytes: int
    processing_status: Literal["PENDING"] = "PENDING"

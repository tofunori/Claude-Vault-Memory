from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryForgetRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=80)
    reason: str = Field(default="manual_admin_forget", max_length=500)
    actor: str = Field(default="admin", max_length=80)


class MemoryRestoreRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=80)
    actor: str = Field(default="admin", max_length=80)


class MemoryUpsertRequest(BaseModel):
    fact_text: str = Field(min_length=1, max_length=600)
    fact_type: Literal["decision", "preference", "constraint", "fact"]
    scope: Literal["global_profile", "working_memory"] = "global_profile"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    actor: str = Field(default="admin", max_length=80)


class ProfileCompactRequest(BaseModel):
    actor: str = Field(default="admin", max_length=80)


class RelationCompactRequest(BaseModel):
    actor: str = Field(default="admin", max_length=80)


class AdminActionResponse(BaseModel):
    status: Literal["queued", "error"]
    ticket_id: str
    message: str


class AdminProfileResponse(BaseModel):
    status: Literal["ok", "error"]
    profile: dict[str, Any] | None = None
    message: str | None = None


class AdminRelationStatsResponse(BaseModel):
    status: Literal["ok", "error"]
    stats: dict[str, Any] | None = None
    message: str | None = None

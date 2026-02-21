from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


AgentName = Literal["claude", "codex", "openclaw"]
EventType = Literal[
    "session_stop",
    "note_add",
    "turn",
    "memory_forget",
    "memory_restore",
    "memory_upsert",
    "profile_compact",
    "relation_compact",
]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(min_length=1, max_length=200)
    agent: AgentName
    top_k: int = Field(default=5, ge=1, le=20)


class NoteMatch(BaseModel):
    note_id: str
    description: str
    type: str
    score: float | None = None
    confidence: str | None = None


class RetrieveDiagnostics(BaseModel):
    latency_ms: int
    search_mode: str
    cache: Literal["ok", "miss"]
    experimental_used: int = 0


class RetrieveResponse(BaseModel):
    status: Literal["ok", "error"]
    context_text: str
    primary: list[NoteMatch]
    linked: list[NoteMatch]
    diagnostics: RetrieveDiagnostics
    message: str | None = None


class EventPayload(BaseModel):
    turn_count: int | None = Field(default=None, ge=0)
    cwd: str | None = None
    conversation_text: str | None = None
    note_id: str | None = None
    note_content: str | None = None
    turn_index: int | None = Field(default=None, ge=1)
    role: Literal["user", "assistant"] | None = None
    text: str | None = Field(default=None, max_length=4000)
    ts: str | None = None
    memory_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)
    actor: str | None = Field(default=None, max_length=80)
    fact_text: str | None = Field(default=None, max_length=600)
    fact_type: Literal["decision", "preference", "constraint", "fact"] | None = None
    scope: Literal["global_profile", "working_memory"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EventRequest(BaseModel):
    event_type: EventType
    agent: AgentName
    session_id: str = Field(min_length=1, max_length=200)
    payload: EventPayload

    @model_validator(mode="after")
    def validate_payload_requirements(self):
        if self.event_type == "session_stop" and not self.payload.conversation_text:
            raise ValueError("payload.conversation_text is required for event_type=session_stop")
        if self.event_type == "note_add":
            if not self.payload.note_id:
                raise ValueError("payload.note_id is required for event_type=note_add")
            if not self.payload.note_content:
                raise ValueError("payload.note_content is required for event_type=note_add")
        if self.event_type == "memory_forget":
            if not self.payload.memory_id:
                raise ValueError("payload.memory_id is required for event_type=memory_forget")
        if self.event_type == "memory_restore":
            if not self.payload.memory_id:
                raise ValueError("payload.memory_id is required for event_type=memory_restore")
        if self.event_type == "memory_upsert":
            if not self.payload.fact_text:
                raise ValueError("payload.fact_text is required for event_type=memory_upsert")
            if not self.payload.fact_type:
                raise ValueError("payload.fact_type is required for event_type=memory_upsert")
            if not self.payload.scope:
                raise ValueError("payload.scope is required for event_type=memory_upsert")
        return self


class EventResponse(BaseModel):
    status: Literal["queued", "error"]
    ticket_id: str
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    queue_db: dict[str, Any]
    worker: dict[str, Any]
    storage: dict[str, Any]
    settings: dict[str, Any]


class AdminReindexResponse(BaseModel):
    status: Literal["ok", "error"]
    message: str
    pid: int | None = None

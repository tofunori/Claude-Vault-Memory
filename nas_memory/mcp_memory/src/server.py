#!/usr/bin/env python3
"""memory MCP server - thin bridge to NAS memory-api."""

import json
import os
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Local .env for this MCP server (optional), then NAS memory .env fallback.
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv("/volume1/Services/memory/.env", override=False)

MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://127.0.0.1:8876").rstrip("/")
MEMORY_API_TOKEN = os.getenv("MEMORY_API_TOKEN", "")
MEMORY_MCP_TIMEOUT_S = float(os.getenv("MEMORY_MCP_TIMEOUT_S", "20"))

mcp = FastMCP("memory")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MEMORY_API_TOKEN:
        headers["Authorization"] = f"Bearer {MEMORY_API_TOKEN}"
    return headers


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{MEMORY_API_URL}{path}",
            json=payload,
            headers=_headers(),
            timeout=MEMORY_MCP_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return {"status": "error", "message": f"memory-api unreachable: {exc}"}

    try:
        data = response.json()
    except ValueError:
        data = {"status": "error", "message": response.text}

    if response.status_code >= 400 and data.get("status") != "error":
        data["status"] = "error"
        data["message"] = data.get("message", f"HTTP {response.status_code}")
    return data


def _enqueue_event_core(
    *,
    event_type: Literal["session_stop", "note_add", "turn"],
    session_id: str,
    payload: dict[str, Any],
    agent: Literal["claude", "codex", "openclaw"] = "codex",
) -> str:
    request_payload = {
        "event_type": event_type,
        "agent": agent,
        "session_id": session_id,
        "payload": payload,
    }
    result = _post("/events", request_payload)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_retrieve(
    query: str,
    session_id: str,
    agent: Literal["claude", "codex", "openclaw"] = "codex",
    top_k: int = 5,
) -> str:
    """
    Retrieve relevant memory context from NAS.

    query: user request or question to contextualize
    session_id: session identifier for traceability
    agent: caller identity (claude|codex|openclaw)
    top_k: number of primary notes to return
    """
    payload = {
        "query": query,
        "session_id": session_id,
        "agent": agent,
        "top_k": top_k,
    }
    result = _post("/retrieve", payload)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_enqueue_event(
    event_type: Literal["session_stop", "note_add", "turn"],
    session_id: str,
    payload: dict[str, Any],
    agent: Literal["claude", "codex", "openclaw"] = "codex",
) -> str:
    """
    Enqueue a memory event in NAS queue.

    event_type: session_stop|note_add|turn
    session_id: logical session id
    payload: event payload object (note_add needs note_id + note_content)
    agent: caller identity (claude|codex|openclaw)
    """
    return _enqueue_event_core(
        event_type=event_type,
        session_id=session_id,
        payload=payload,
        agent=agent,
    )


@mcp.tool()
def memory_note_add(
    session_id: str,
    note_id: str,
    note_content: str,
    agent: Literal["claude", "codex", "openclaw"] = "codex",
) -> str:
    """
    Convenience wrapper: enqueue a note_add event.

    session_id: logical session id
    note_id: note filename without .md
    note_content: full markdown note content
    agent: caller identity (claude|codex|openclaw)
    """
    return _enqueue_event_core(
        event_type="note_add",
        session_id=session_id,
        payload={"note_id": note_id, "note_content": note_content},
        agent=agent,
    )


@mcp.tool()
def memory_health() -> str:
    """
    Get memory-api health diagnostics.
    """
    try:
        response = requests.get(
            f"{MEMORY_API_URL}/health",
            headers=_headers(),
            timeout=MEMORY_MCP_TIMEOUT_S,
        )
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        data = {"status": "error", "message": str(exc)}
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_admin_forget(memory_id: str, reason: str = "manual_admin_forget", actor: str = "admin") -> str:
    """
    Admin helper: request soft-forget for one memory node by id.
    """
    result = _post("/admin/memory/forget", {"memory_id": memory_id, "reason": reason, "actor": actor})
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_admin_restore(memory_id: str, actor: str = "admin") -> str:
    """
    Admin helper: restore a previously forgotten memory node by id.
    """
    result = _post("/admin/memory/restore", {"memory_id": memory_id, "actor": actor})
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_admin_upsert(
    fact_text: str,
    fact_type: Literal["decision", "preference", "constraint", "fact"] = "fact",
    scope: Literal["global_profile", "working_memory"] = "global_profile",
    confidence: float = 0.5,
    actor: str = "admin",
) -> str:
    """
    Admin helper: upsert one memory fact in versioned memory nodes.
    """
    result = _post(
        "/admin/memory/upsert",
        {
            "fact_text": fact_text,
            "fact_type": fact_type,
            "scope": scope,
            "confidence": confidence,
            "actor": actor,
        },
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_admin_profile() -> str:
    """
    Admin helper: fetch latest global profile snapshot.
    """
    try:
        response = requests.get(
            f"{MEMORY_API_URL}/admin/profile",
            headers=_headers(),
            timeout=MEMORY_MCP_TIMEOUT_S,
        )
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        data = {"status": "error", "message": str(exc)}
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()

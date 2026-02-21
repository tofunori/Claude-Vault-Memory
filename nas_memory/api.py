from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import subprocess
import time
import uuid
import fcntl
from pathlib import Path

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .admin_models import (
    AdminActionResponse,
    AdminProfileResponse,
    AdminRelationStatsResponse,
    MemoryForgetRequest,
    MemoryRestoreRequest,
    MemoryUpsertRequest,
    ProfileCompactRequest,
    RelationCompactRequest,
)
from .config import Settings, load_settings
from .db import (
    cleanup_expired_staging,
    enqueue_event,
    get_latest_profile_snapshot,
    get_memory_node_counts,
    get_relation_stats,
    get_stats,
    init_db,
    list_active_memory_nodes,
    list_staging_for_session,
)
from .locks import exclusive_lock
from .models import (
    AdminReindexResponse,
    EventRequest,
    EventResponse,
    HealthResponse,
    NoteMatch,
    RetrieveDiagnostics,
    RetrieveRequest,
    RetrieveResponse,
)
from .security import verify_request_security
from .graph_view import build_unified_graph

app = FastAPI(title="NAS Memory API", version="1.0.0")
_SETTINGS: Settings = load_settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s nas-memory-api %(levelname)s: %(message)s")
_LOG = logging.getLogger("uvicorn.error")


def _log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    _LOG.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _worker_lock_held(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    fd = None
    try:
        fd = lock_path.open("a+")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    except Exception:
        return False
    finally:
        if fd is not None:
            fd.close()


def _build_context_input(query: str, session_id: str) -> str:
    payload = {"prompt": query, "session_id": session_id}
    return json.dumps(payload, ensure_ascii=False)


def _parse_note_line(line: str) -> NoteMatch | None:
    m = re.match(r"^\[\[([^\]]+)\]\]\s+\(([^)]*)\)\s+—\s*(.*)$", line.strip())
    if not m:
        return None

    note_id, meta, description = m.groups()
    score = None
    confidence = "experimental"
    note_type = "?"

    parts = [p.strip() for p in meta.split(",") if p.strip()]
    if parts:
        note_type = parts[0]

    percent_match = re.search(r"(\d+)%", meta)
    if percent_match:
        score = int(percent_match.group(1)) / 100.0

    if "[confirmed]" in meta:
        confidence = "confirmed"

    return NoteMatch(
        note_id=note_id,
        description=description,
        type=note_type,
        score=score,
        confidence=confidence,
    )


def _staging_to_note(item: dict, score_cap: float) -> NoteMatch:
    confidence = float(item.get("confidence", 0.0))
    score = min(score_cap, max(0.0, confidence))
    return NoteMatch(
        note_id=f"staging-{str(item.get('id', 'item'))[:8]}",
        description=str(item.get("fact_text", "")),
        type=f"staging/{item.get('fact_type', 'fact')}",
        score=score,
        confidence="experimental",
    )


def _profile_to_note(item: dict, score_cap: float) -> NoteMatch:
    confidence = float(item.get("confidence", 0.0))
    score = min(score_cap, max(0.0, confidence))
    return NoteMatch(
        note_id=f"profile-{str(item.get('id', 'item'))[:8]}",
        description=str(item.get("fact_text", "")),
        type=f"profile/{item.get('fact_type', 'fact')}",
        score=score,
        confidence="confirmed",
    )


def _parse_retrieve_output(text: str, top_k: int) -> tuple[list[NoteMatch], list[NoteMatch]]:
    primary: list[NoteMatch] = []
    linked: list[NoteMatch] = []
    section = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("=== Relevant vault notes"):
            section = "primary"
            continue
        if line.startswith("=== Connected notes"):
            section = "linked"
            continue
        if line.startswith("=== Source context"):
            section = "source"
            continue
        if line.startswith("==="):
            section = ""
            continue

        note = _parse_note_line(line)
        if not note:
            continue

        if section == "primary":
            primary.append(note)
        elif section == "linked":
            linked.append(note)

    return primary[:top_k], linked


def _retrieve_via_cli(req: RetrieveRequest) -> tuple[str, list[NoteMatch], list[NoteMatch]]:
    if not _SETTINGS.vault_retrieve_script.exists():
        raise RuntimeError(f"Missing script: {_SETTINGS.vault_retrieve_script}")

    with exclusive_lock(_SETTINGS.qdrant_lock_path, _SETTINGS.qdrant_lock_timeout_seconds):
        proc = subprocess.run(
            ["python3", str(_SETTINGS.vault_retrieve_script)],
            input=_build_context_input(req.query, req.session_id),
            text=True,
            capture_output=True,
            cwd=str(_SETTINGS.repo_root),
            timeout=_SETTINGS.retrieve_timeout_seconds,
        )

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(stderr[:500] if stderr else f"vault_retrieve exited with {proc.returncode}")

    context_text = proc.stdout.strip()
    primary, linked = _parse_retrieve_output(context_text, req.top_k)
    return context_text, primary, linked


def _dedup_hash(event_type: str, agent: str, session_id: str, payload: dict) -> str:
    canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    src = f"{event_type}|{agent}|{session_id}|{canonical_payload}".encode("utf-8")
    return hashlib.sha256(src).hexdigest()


def _enqueue_event_or_duplicate(
    *,
    event_type: str,
    agent: str,
    session_id: str,
    payload: dict,
) -> tuple[str, str]:
    ticket_id = str(uuid.uuid4())
    dedup_hash = _dedup_hash(event_type, agent, session_id, payload)
    inserted, stored_id = enqueue_event(
        _SETTINGS.queue_db_path,
        event_id=ticket_id,
        event_type=event_type,
        agent=agent,
        session_id=session_id,
        payload=payload,
        dedup_hash=dedup_hash,
    )
    status = "queued" if inserted else "duplicate"
    return status, stored_id


@app.on_event("startup")
def on_startup() -> None:
    init_db(_SETTINGS.queue_db_path)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    _: None = Depends(verify_request_security),
):
    t0 = time.perf_counter()
    retrieve_status = "error"
    experimental_used = 0
    profile_used = 0

    try:
        context_text, primary, linked = _retrieve_via_cli(req)
        cleanup_expired_staging(_SETTINGS.queue_db_path)
        experimental = list_staging_for_session(
            _SETTINGS.queue_db_path,
            session_id=req.session_id,
            statuses=("experimental",),
            limit=_SETTINGS.retrieve_experimental_max,
        )
        experimental_notes = [
            _staging_to_note(item, _SETTINGS.retrieve_experimental_score_cap)
            for item in experimental
        ]
        experimental_used = len(experimental_notes)
        if experimental_notes:
            linked.extend(experimental_notes)
            block_lines = ["=== Session experimental notes ==="]
            for note in experimental_notes:
                pct = int(round((note.score or 0.0) * 100))
                block_lines.append(
                    f"[[{note.note_id}]] ({note.type}, {pct}% [experimental]) — {note.description}"
                )
            context_text = (context_text + "\n\n" + "\n".join(block_lines)).strip()

        if _SETTINGS.profile_enable:
            profile_rows = list_active_memory_nodes(
                _SETTINGS.queue_db_path,
                scope="global_profile",
                limit=_SETTINGS.profile_max_items,
            )
            profile_notes = [
                _profile_to_note(item, _SETTINGS.profile_score_cap)
                for item in profile_rows[: _SETTINGS.profile_max_items]
            ]
            profile_used = len(profile_notes)
            if profile_notes:
                linked.extend(profile_notes)
                block_lines = ["=== Global profile memory ==="]
                for note in profile_notes:
                    pct = int(round((note.score or 0.0) * 100))
                    block_lines.append(
                        f"[[{note.note_id}]] ({note.type}, {pct}% [confirmed]) — {note.description}"
                    )
                context_text = (context_text + "\n\n" + "\n".join(block_lines)).strip()

        latency_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics = RetrieveDiagnostics(
            latency_ms=latency_ms,
            search_mode=_SETTINGS.search_mode,
            cache="ok" if _SETTINGS.graph_cache_path.exists() else "miss",
            experimental_used=experimental_used,
        )
        retrieve_status = "ok"
        return RetrieveResponse(
            status="ok",
            context_text=context_text,
            primary=primary,
            linked=linked,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics = RetrieveDiagnostics(
            latency_ms=latency_ms,
            search_mode=_SETTINGS.search_mode,
            cache="ok" if _SETTINGS.graph_cache_path.exists() else "miss",
            experimental_used=0,
        )
        return RetrieveResponse(
            status="error",
            context_text="",
            primary=[],
            linked=[],
            diagnostics=diagnostics,
            message=str(exc),
        )
    finally:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _log_event(
            "retrieve",
            status=retrieve_status,
            session_id=req.session_id,
            agent=req.agent,
            top_k=req.top_k,
            latency_ms=latency_ms,
            search_mode=_SETTINGS.search_mode,
            experimental_used=experimental_used,
            profile_used=profile_used,
        )


@app.post("/events", response_model=EventResponse)
def events(
    req: EventRequest,
    _: None = Depends(verify_request_security),
):
    payload = req.payload.model_dump(exclude_none=True)
    if req.event_type == "turn":
        missing: list[str] = []
        if payload.get("turn_index") is None:
            missing.append("turn_index")
        if not payload.get("role"):
            missing.append("role")
        if not payload.get("text"):
            missing.append("text")
        if missing:
            _log_event(
                "events",
                status="error",
                ticket_id="invalid",
                event_type=req.event_type,
                agent=req.agent,
                session_id=req.session_id,
                message=f"missing:{','.join(missing)}",
            )
            return EventResponse(
                status="error",
                ticket_id="invalid",
                message=f"turn payload missing required fields: {', '.join(missing)}",
            )

    status, stored_id = _enqueue_event_or_duplicate(
        event_type=req.event_type,
        agent=req.agent,
        session_id=req.session_id,
        payload=payload,
    )

    if status == "queued":
        _log_event(
            "events",
            status="queued",
            ticket_id=stored_id,
            event_type=req.event_type,
            agent=req.agent,
            session_id=req.session_id,
        )
        return EventResponse(status="queued", ticket_id=stored_id, message="Event queued")

    _log_event(
        "events",
        status="duplicate",
        ticket_id=stored_id,
        event_type=req.event_type,
        agent=req.agent,
        session_id=req.session_id,
    )
    return EventResponse(status="queued", ticket_id=stored_id, message="Duplicate ignored (idempotent)")


def _require_profile_enabled() -> None:
    if not _SETTINGS.profile_enable:
        raise RuntimeError("profile layer is disabled (MEMORY_PROFILE_ENABLE=false)")


def _admin_graph_ui_path() -> Path:
    return _SETTINGS.repo_root / "nas_memory" / "static" / "graph.html"


def _build_graph_with_timeout() -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(build_unified_graph, _SETTINGS)
        return future.result(timeout=_SETTINGS.admin_graph_timeout_seconds)


@app.post("/admin/memory/forget", response_model=AdminActionResponse)
def admin_memory_forget(
    req: MemoryForgetRequest,
    _: None = Depends(verify_request_security),
):
    try:
        _require_profile_enabled()
        payload = {"memory_id": req.memory_id, "reason": req.reason, "actor": req.actor}
        status, ticket_id = _enqueue_event_or_duplicate(
            event_type="memory_forget",
            agent="codex",
            session_id=f"admin-forget-{req.memory_id}",
            payload=payload,
        )
        _log_event("admin_memory_forget", status=status, ticket_id=ticket_id, memory_id=req.memory_id)
        return AdminActionResponse(
            status="queued",
            ticket_id=ticket_id,
            message="Event queued" if status == "queued" else "Duplicate ignored (idempotent)",
        )
    except Exception as exc:
        return AdminActionResponse(status="error", ticket_id="error", message=str(exc))


@app.post("/admin/memory/restore", response_model=AdminActionResponse)
def admin_memory_restore(
    req: MemoryRestoreRequest,
    _: None = Depends(verify_request_security),
):
    try:
        _require_profile_enabled()
        payload = {"memory_id": req.memory_id, "actor": req.actor}
        status, ticket_id = _enqueue_event_or_duplicate(
            event_type="memory_restore",
            agent="codex",
            session_id=f"admin-restore-{req.memory_id}",
            payload=payload,
        )
        _log_event("admin_memory_restore", status=status, ticket_id=ticket_id, memory_id=req.memory_id)
        return AdminActionResponse(
            status="queued",
            ticket_id=ticket_id,
            message="Event queued" if status == "queued" else "Duplicate ignored (idempotent)",
        )
    except Exception as exc:
        return AdminActionResponse(status="error", ticket_id="error", message=str(exc))


@app.post("/admin/memory/upsert", response_model=AdminActionResponse)
def admin_memory_upsert(
    req: MemoryUpsertRequest,
    _: None = Depends(verify_request_security),
):
    try:
        _require_profile_enabled()
        payload = {
            "fact_text": req.fact_text,
            "fact_type": req.fact_type,
            "scope": req.scope,
            "confidence": req.confidence,
            "actor": req.actor,
        }
        session_seed = f"{req.scope}:{req.fact_type}:{req.fact_text[:48]}"
        status, ticket_id = _enqueue_event_or_duplicate(
            event_type="memory_upsert",
            agent="codex",
            session_id=f"admin-upsert-{hashlib.sha1(session_seed.encode('utf-8')).hexdigest()[:16]}",
            payload=payload,
        )
        _log_event("admin_memory_upsert", status=status, ticket_id=ticket_id, scope=req.scope, fact_type=req.fact_type)
        return AdminActionResponse(
            status="queued",
            ticket_id=ticket_id,
            message="Event queued" if status == "queued" else "Duplicate ignored (idempotent)",
        )
    except Exception as exc:
        return AdminActionResponse(status="error", ticket_id="error", message=str(exc))


@app.post("/admin/profile/compact", response_model=AdminActionResponse)
def admin_profile_compact(
    req: ProfileCompactRequest,
    _: None = Depends(verify_request_security),
):
    try:
        _require_profile_enabled()
        payload = {"actor": req.actor}
        status, ticket_id = _enqueue_event_or_duplicate(
            event_type="profile_compact",
            agent="codex",
            session_id=f"admin-compact-{int(time.time() // 60)}",
            payload=payload,
        )
        _log_event("admin_profile_compact", status=status, ticket_id=ticket_id)
        return AdminActionResponse(
            status="queued",
            ticket_id=ticket_id,
            message="Event queued" if status == "queued" else "Duplicate ignored (idempotent)",
        )
    except Exception as exc:
        return AdminActionResponse(status="error", ticket_id="error", message=str(exc))


@app.post("/admin/relations/compact", response_model=AdminActionResponse)
def admin_relations_compact(
    req: RelationCompactRequest,
    _: None = Depends(verify_request_security),
):
    try:
        payload = {"actor": req.actor}
        interval_seconds = max(60, int(_SETTINGS.relation_compact_interval_min) * 60)
        bucket = int(time.time() // interval_seconds)
        status, ticket_id = _enqueue_event_or_duplicate(
            event_type="relation_compact",
            agent="codex",
            session_id=f"admin-relations-compact-{bucket}",
            payload=payload,
        )
        _log_event("admin_relations_compact", status=status, ticket_id=ticket_id)
        return AdminActionResponse(
            status="queued",
            ticket_id=ticket_id,
            message="Event queued" if status == "queued" else "Duplicate ignored (idempotent)",
        )
    except Exception as exc:
        return AdminActionResponse(status="error", ticket_id="error", message=str(exc))


@app.get("/admin/relations/stats", response_model=AdminRelationStatsResponse)
def admin_relations_stats(
    _: None = Depends(verify_request_security),
):
    try:
        stats = get_relation_stats(_SETTINGS.queue_db_path)
        return AdminRelationStatsResponse(status="ok", stats=stats)
    except Exception as exc:
        return AdminRelationStatsResponse(status="error", stats=None, message=str(exc))


@app.get("/admin/profile", response_model=AdminProfileResponse)
def admin_profile(
    _: None = Depends(verify_request_security),
):
    try:
        _require_profile_enabled()
        profile = get_latest_profile_snapshot(_SETTINGS.queue_db_path, profile_name="global")
        return AdminProfileResponse(status="ok", profile=profile or {})
    except Exception as exc:
        return AdminProfileResponse(status="error", profile=None, message=str(exc))


@app.get("/admin/graph")
def admin_graph(
    _: None = Depends(verify_request_security),
):
    t0 = time.perf_counter()
    try:
        payload = _build_graph_with_timeout()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        stats = payload.get("stats", {})
        _log_event(
            "admin_graph",
            status="ok",
            latency_ms=latency_ms,
            node_count=len(payload.get("nodes", [])),
            edge_count=len(payload.get("edges", [])),
            warning_count=len(payload.get("warnings", [])),
            note_nodes=stats.get("note_nodes", 0),
            memory_nodes=stats.get("memory_nodes", 0),
            components=stats.get("components", 0),
        )
        return payload
    except concurrent.futures.TimeoutError:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _log_event(
            "admin_graph",
            status="error",
            latency_ms=latency_ms,
            message="graph build timeout",
            timeout_seconds=_SETTINGS.admin_graph_timeout_seconds,
        )
        return {
            "status": "error",
            "generated_at": None,
            "stats": {
                "note_nodes": 0,
                "memory_nodes": 0,
                "bridge_edges": 0,
                "note_edges": 0,
                "memory_edges": 0,
                "components": 0,
            },
            "nodes": [],
            "edges": [],
            "warnings": [
                f"graph build timed out after {_SETTINGS.admin_graph_timeout_seconds}s"
            ],
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _log_event("admin_graph", status="error", latency_ms=latency_ms, message=str(exc))
        return {
            "status": "error",
            "generated_at": None,
            "stats": {
                "note_nodes": 0,
                "memory_nodes": 0,
                "bridge_edges": 0,
                "note_edges": 0,
                "memory_edges": 0,
                "components": 0,
            },
            "nodes": [],
            "edges": [],
            "warnings": [str(exc)],
        }


@app.get("/admin/graph/ui", response_class=HTMLResponse)
def admin_graph_ui(
    request: Request,
    authorization: str | None = Header(default=None),
    token: str | None = None,
):
    auth_header = authorization
    if not auth_header and token:
        auth_header = f"Bearer {token}"
    verify_request_security(request, authorization=auth_header)

    ui_path = _admin_graph_ui_path()
    if not ui_path.exists():
        return HTMLResponse(
            status_code=500,
            content="<h1>Graph UI missing</h1><p>Expected file nas_memory/static/graph.html</p>",
        )
    return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health(
    _: None = Depends(verify_request_security),
):
    queue_stats = get_stats(_SETTINGS.queue_db_path)
    memory_counts = get_memory_node_counts(_SETTINGS.queue_db_path)
    relation_stats = get_relation_stats(_SETTINGS.queue_db_path)
    latest_profile = get_latest_profile_snapshot(_SETTINGS.queue_db_path, profile_name="global")
    lock_held = _worker_lock_held(_SETTINGS.worker_lock_path)

    worker_info = {
        "lock_path": str(_SETTINGS.worker_lock_path),
        "lock_exists": _SETTINGS.worker_lock_path.exists(),
        "lock_held": lock_held,
    }
    if _SETTINGS.worker_lock_path.exists():
        try:
            worker_info["lock_content"] = _SETTINGS.worker_lock_path.read_text(encoding="utf-8").strip()
        except Exception:
            worker_info["lock_content"] = ""

    storage = {
        "vault_notes_dir": str(_SETTINGS.vault_notes_dir),
        "vault_notes_exists": _SETTINGS.vault_notes_dir.exists(),
        "qdrant_path": str(_SETTINGS.qdrant_path),
        "qdrant_exists": _SETTINGS.qdrant_path.exists(),
        "bm25_index_path": str(_SETTINGS.bm25_index_path),
        "bm25_index_exists": _SETTINGS.bm25_index_path.exists(),
        "graph_cache_path": str(_SETTINGS.graph_cache_path),
        "graph_cache_exists": _SETTINGS.graph_cache_path.exists(),
        "profile_last_version": int(latest_profile.get("version", 0)) if latest_profile else 0,
        "profile_last_compacted_at": latest_profile.get("created_at") if latest_profile else None,
        "memory_nodes_active_count": memory_counts.get("active", 0),
        "memory_nodes_forgotten_count": memory_counts.get("forgotten", 0),
        "memory_singletons": relation_stats.get("memory_singletons", 0),
        "relation_edges_total": relation_stats.get("memory_edges_total", 0),
        "relation_last_candidate_at": relation_stats.get("last_candidate_at"),
    }

    settings_view = {
        "repo_root": str(_SETTINGS.repo_root),
        "core_config_loaded": _SETTINGS.core_config_loaded,
        "search_mode": _SETTINGS.search_mode,
        "allowed_ips": list(_SETTINGS.allowed_ips),
        "turn_live_cadence": _SETTINGS.turn_live_cadence,
        "staging_ttl_hours": _SETTINGS.staging_ttl_hours,
        "retrieve_experimental_max": _SETTINGS.retrieve_experimental_max,
        "profile_enable": _SETTINGS.profile_enable,
        "profile_max_items": _SETTINGS.profile_max_items,
        "profile_score_cap": _SETTINGS.profile_score_cap,
        "admin_graph_timeout_seconds": _SETTINGS.admin_graph_timeout_seconds,
        "relation_enable": _SETTINGS.relation_enable,
        "relation_write": _SETTINGS.relation_write,
        "relation_batch_max_pairs": _SETTINGS.relation_batch_max_pairs,
        "relation_min_confidence": _SETTINGS.relation_min_confidence,
        "relation_max_new_edges_per_run": _SETTINGS.relation_max_new_edges_per_run,
        "relation_compact_interval_min": _SETTINGS.relation_compact_interval_min,
        "relation_llm_timeout": _SETTINGS.relation_llm_timeout,
    }

    degraded = not storage["vault_notes_exists"] or not storage["qdrant_exists"]
    service_status = "degraded" if degraded else "ok"

    return HealthResponse(
        status=service_status,
        service="nas-memory-api",
        queue_db={"path": str(_SETTINGS.queue_db_path), **queue_stats},
        worker=worker_info,
        storage=storage,
        settings=settings_view,
    )


@app.post("/admin/reindex", response_model=AdminReindexResponse)
def admin_reindex(
    _: None = Depends(verify_request_security),
):
    if not _SETTINGS.vault_embed_script.exists():
        return JSONResponse(
            status_code=500,
            content=AdminReindexResponse(
                status="error",
                message=f"Missing script: {_SETTINGS.vault_embed_script}",
            ).model_dump(),
        )

    try:
        proc = subprocess.Popen(
            ["python3", str(_SETTINGS.vault_embed_script)],
            cwd=str(_SETTINGS.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        _log_event("admin_reindex", status="error", message=str(exc))
        return JSONResponse(
            status_code=500,
            content=AdminReindexResponse(status="error", message=str(exc)).model_dump(),
        )

    _log_event("admin_reindex", status="started", pid=proc.pid)
    return AdminReindexResponse(
        status="ok",
        message="Reindex started",
        pid=proc.pid,
    )

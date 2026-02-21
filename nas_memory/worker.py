from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import load_settings
from .db import (
    add_staging_evidence,
    append_memory_audit,
    claim_next_event,
    cleanup_expired_staging,
    create_profile_snapshot,
    delete_session_buffer,
    forget_memory_node,
    get_stats,
    init_db,
    list_active_memory_nodes,
    list_all_active_memory_nodes,
    list_recent_active_memory_nodes,
    list_staging_for_session,
    mark_done,
    mark_error,
    insert_memory_edge_if_missing,
    restore_memory_node,
    set_staging_status_for_ids,
    upsert_relation_candidate,
    upsert_memory_node_versioned,
    upsert_session_buffer,
    upsert_staging_memory,
)
from .locks import exclusive_lock
from .relation_linker import generate_relation_candidates

TODAY = date.today().isoformat()


def log(msg: str, **fields) -> None:
    payload = {"date": TODAY, "message": msg, **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def sanitize_note_id(note_id: str) -> str:
    note_id = note_id.lower().strip()
    note_id = re.sub(r"[^a-z0-9\-]", "-", note_id)
    note_id = re.sub(r"-+", "-", note_id)
    return note_id.strip("-")[:80] or "note"


def write_file_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp).replace(path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)


def _canonical_fact_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 \-_/.:]", "", text)
    return text[:300]


def _fingerprint(fact_text: str, fact_type: str) -> str:
    src = f"{fact_type}|{_canonical_fact_text(fact_text)}".encode("utf-8")
    return hashlib.sha256(src).hexdigest()


def _memory_global_key(scope: str, fact_type: str, fact_text: str) -> str:
    canonical = _canonical_fact_text(fact_text)
    tokens = [t for t in re.split(r"[^a-z0-9]+", canonical) if t]
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "on",
        "in",
        "for",
        "with",
        "is",
        "are",
        "be",
        "we",
        "use",
        "using",
        "on",
        "not",
        "no",
        "never",
        "pas",
        "ne",
        "sans",
        "de",
        "la",
        "le",
        "les",
        "un",
        "une",
        "des",
        "et",
        "ou",
        "que",
        "qui",
        "on",
    }
    core = [t for t in tokens if t not in stop]
    if not core:
        core = tokens
    topic = " ".join(core[:8])[:180]
    src = f"{scope}|{fact_type}|{topic}".encode("utf-8")
    return hashlib.sha256(src).hexdigest()


def _relation_mode(existing_text: str, new_text: str) -> str:
    old = _canonical_fact_text(existing_text)
    new = _canonical_fact_text(new_text)
    if old == new:
        return "same"

    neg_terms = (
        " not ",
        " never ",
        " no ",
        " pas ",
        " ne ",
        " sans ",
        " interdit ",
        " impossible ",
    )
    old_neg = any(term in f" {old} " for term in neg_terms)
    new_neg = any(term in f" {new} " for term in neg_terms)
    if old_neg != new_neg:
        return "contradicts"
    return "updates"


def _has_explicit_confirmation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(oui c'?est valid[ée]|on garde [cç]a|d[ée]cision finale|this is final|approved|confirmed)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _run_live_extract(settings, session_id: str, agent: str, turns: list[dict]) -> dict:
    if not settings.live_extract_script.exists():
        return {"candidates": [], "error": f"missing script {settings.live_extract_script}"}

    payload = {
        "session_id": session_id,
        "agent": agent,
        "turns": turns,
        "max_candidates": settings.live_extract_max_candidates,
        "timeout_s": settings.live_extract_timeout_seconds,
    }
    proc = _run_subprocess(
        ["python3", str(settings.live_extract_script)],
        cwd=settings.repo_root,
        timeout=max(2, settings.live_extract_timeout_seconds + 2),
        input_text=json.dumps(payload, ensure_ascii=False),
    )
    if proc.returncode != 0:
        return {"candidates": [], "error": f"live_extract failed rc={proc.returncode}: {proc.stderr[:180]}"}

    try:
        parsed = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {"candidates": [], "error": f"live_extract json parse error: {exc}"}

    if not isinstance(parsed, dict):
        return {"candidates": [], "error": "live_extract invalid payload"}
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        parsed["candidates"] = []
    return parsed


def _run_profile_extract(settings, *, conversation_text: str, staging_items: list[dict]) -> dict:
    if not settings.profile_extract_script.exists():
        return {"static": [], "dynamic": [], "error": f"missing script {settings.profile_extract_script}"}
    payload = {
        "conversation_text": conversation_text,
        "staging_items": staging_items,
        "max_items": settings.profile_max_items,
        "timeout_s": settings.profile_extract_timeout_seconds,
    }
    proc = _run_subprocess(
        ["python3", str(settings.profile_extract_script)],
        cwd=settings.repo_root,
        timeout=max(2, settings.profile_extract_timeout_seconds + 3),
        input_text=json.dumps(payload, ensure_ascii=False),
    )
    if proc.returncode != 0:
        return {"static": [], "dynamic": [], "error": f"profile_extract failed rc={proc.returncode}: {proc.stderr[:180]}"}
    try:
        parsed = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {"static": [], "dynamic": [], "error": f"profile_extract json parse error: {exc}"}
    if not isinstance(parsed, dict):
        return {"static": [], "dynamic": [], "error": "profile_extract invalid payload"}
    if not isinstance(parsed.get("static"), list):
        parsed["static"] = []
    if not isinstance(parsed.get("dynamic"), list):
        parsed["dynamic"] = []
    return parsed


def _format_staging_block(staging_items: list[dict]) -> str:
    lines = ["[STAGING MEMORY CANDIDATES - FOR CONSOLIDATION ONLY]"]
    for s in staging_items:
        lines.append(
            "- {status} | {fact_type} | evidence={evidence_count} | conf={confidence:.2f}: {fact_text}".format(
                status=s.get("status", "experimental"),
                fact_type=s.get("fact_type", "fact"),
                evidence_count=int(s.get("evidence_count", 1)),
                confidence=float(s.get("confidence", 0.0)),
                fact_text=str(s.get("fact_text", ""))[:300],
            )
        )
    return "\n".join(lines)


def _conversation_to_events(conversation_text: str) -> list[dict]:
    lines = conversation_text.splitlines()
    has_explicit_roles = any(line.startswith("USER:") or line.startswith("ASSISTANT:") for line in lines)

    if not has_explicit_roles:
        return [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": conversation_text,
                },
            }
        ]

    events: list[dict] = []
    current_role = None
    buffer: list[str] = []

    def flush_buffer():
        nonlocal buffer, current_role
        if not current_role:
            return
        text = "\n".join(buffer).strip()
        if not text:
            return
        role = "assistant" if current_role == "ASSISTANT" else "user"
        events.append({"type": role, "message": {"role": role, "content": text}})

    for line in lines:
        if line.startswith("USER:"):
            flush_buffer()
            current_role = "USER"
            buffer = [line[len("USER:"):].strip()]
        elif line.startswith("ASSISTANT:"):
            flush_buffer()
            current_role = "ASSISTANT"
            buffer = [line[len("ASSISTANT:"):].strip()]
        else:
            buffer.append(line)

    flush_buffer()

    if not events:
        events = [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": conversation_text,
                },
            }
        ]

    return events


class WorkerLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, f"pid={os.getpid()}\nstarted={int(time.time())}\n".encode("utf-8"))

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _run_full_reindex(settings) -> None:
    if not settings.rebuild_full_after_write:
        return
    log("Reindex full start")
    proc = _run_subprocess(
        ["python3", str(settings.vault_embed_script)],
        cwd=settings.repo_root,
        timeout=settings.reindex_timeout_seconds,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vault_embed full failed: {proc.stderr[:500]}")
    log("Reindex full done")


def _upsert_memory_item(
    settings,
    *,
    scope: str,
    fact_text: str,
    fact_type: str,
    confidence: float,
    source: str,
    evidence_increment: int,
    actor: str,
) -> dict:
    global_key = _memory_global_key(scope, fact_type, fact_text)
    current = None
    if scope in {"working_memory", "global_profile"}:
        current = list_active_memory_nodes(settings.queue_db_path, scope=scope, limit=200)
        current = next((x for x in current if x.get("global_key") == global_key), None)

    relation = "same"
    if current is not None:
        relation = _relation_mode(str(current.get("fact_text", "")), fact_text)

    result = upsert_memory_node_versioned(
        settings.queue_db_path,
        global_key=global_key,
        scope=scope,
        fact_text=fact_text,
        fact_type=fact_type,
        confidence=confidence,
        source=source,
        relation_mode=relation,
        evidence_increment=max(1, int(evidence_increment)),
    )
    append_memory_audit(
        settings.queue_db_path,
        action="upsert",
        target_id=str(result.get("node_id")),
        payload={
            "scope": scope,
            "fact_type": fact_type,
            "fact_text": fact_text[:220],
            "global_key": global_key,
            "relation": result.get("edge_relation"),
            "from_version": result.get("from_version"),
            "to_version": result.get("to_version"),
            "source": source,
        },
        actor=actor,
    )
    log(
        "memory_action",
        action="upsert",
        global_key=global_key,
        from_version=result.get("from_version"),
        to_version=result.get("to_version"),
        relation=result.get("edge_relation"),
        source=source,
    )
    return result


def _compact_profile_snapshot(
    settings,
    *,
    created_by: str,
    source_window_start: str | None = None,
    source_window_end: str | None = None,
) -> dict:
    static_rows = list_active_memory_nodes(
        settings.queue_db_path, scope="global_profile", limit=settings.profile_max_items
    )
    dynamic_rows = list_recent_active_memory_nodes(
        settings.queue_db_path,
        scope="working_memory",
        updated_after=source_window_start or "1970-01-01T00:00:00+00:00",
        limit=settings.profile_max_items,
    )

    static_items = [
        {
            "memory_id": row.get("id"),
            "fact_text": row.get("fact_text"),
            "fact_type": row.get("fact_type"),
            "confidence": row.get("confidence"),
            "evidence_count": row.get("evidence_count"),
        }
        for row in static_rows
    ]
    dynamic_items = [
        {
            "memory_id": row.get("id"),
            "fact_text": row.get("fact_text"),
            "fact_type": row.get("fact_type"),
            "confidence": row.get("confidence"),
            "evidence_count": row.get("evidence_count"),
        }
        for row in dynamic_rows
    ]

    snapshot = create_profile_snapshot(
        settings.queue_db_path,
        profile_name="global",
        static_items=static_items,
        dynamic_items=dynamic_items,
        created_by=created_by,
        source_window_start=source_window_start,
        source_window_end=source_window_end,
    )
    append_memory_audit(
        settings.queue_db_path,
        action="compact",
        target_id=str(snapshot.get("id")),
        payload={
            "profile_name": "global",
            "version": snapshot.get("version"),
            "static_count": len(static_items),
            "dynamic_count": len(dynamic_items),
            "created_by": created_by,
        },
        actor=created_by,
    )
    log(
        "memory_action",
        action="compact",
        global_key="profile:global",
        from_version=(int(snapshot.get("version", 1)) - 1),
        to_version=snapshot.get("version"),
        static_count=len(static_items),
        dynamic_count=len(dynamic_items),
    )
    return snapshot


def _handle_note_add(settings, payload: dict, event_id: str) -> None:
    raw_note_id = payload.get("note_id", "")
    note_content = payload.get("note_content", "")
    if not raw_note_id or not note_content:
        raise ValueError("note_add missing note_id or note_content")

    note_id = sanitize_note_id(raw_note_id)
    note_path = settings.vault_notes_dir / f"{note_id}.md"
    write_file_atomic(note_path, note_content)
    log("note_add wrote", event_id=event_id, note_id=note_id, note_path=str(note_path))

    with exclusive_lock(settings.qdrant_lock_path, settings.qdrant_lock_timeout_seconds):
        proc = _run_subprocess(
            ["python3", str(settings.vault_embed_script), "--note", note_id],
            cwd=settings.repo_root,
            timeout=settings.embed_timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"vault_embed --note failed: {proc.stderr[:500]}")
        _run_full_reindex(settings)
    log("note_add done", event_id=event_id, note_id=note_id)


def _handle_session_stop(settings, payload: dict, session_id: str, event_id: str) -> None:
    conversation_text = payload.get("conversation_text", "")
    if not conversation_text:
        raise ValueError("session_stop missing conversation_text")

    transcripts_dir = settings.queue_dir / "_remote_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    events = _conversation_to_events(conversation_text)
    staging_items = list_staging_for_session(
        settings.queue_db_path,
        session_id=session_id,
        statuses=("experimental", "confirmed"),
        limit=25,
        only_unexpired=False,
    )
    if staging_items:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": _format_staging_block(staging_items),
                },
            }
        )
    transcript_path = transcripts_dir / f"{session_id}-{event_id}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    turn_count = payload.get("turn_count") or len(events)
    ticket_path = settings.queue_dir / f"{session_id}-{event_id}.json"
    ticket = {
        "session_id": f"{session_id}-{event_id[:8]}",
        "transcript_path": str(transcript_path),
        "cwd": payload.get("cwd", ""),
        "turn_count": turn_count,
        "enqueued_at": TODAY,
    }
    write_file_atomic(ticket_path, json.dumps(ticket, ensure_ascii=False, indent=2))

    with exclusive_lock(settings.qdrant_lock_path, settings.qdrant_lock_timeout_seconds):
        proc = _run_subprocess(
            ["python3", str(settings.process_queue_script)],
            cwd=settings.repo_root,
            timeout=settings.process_queue_timeout_seconds,
            env_overrides={"DISABLE_ASYNC_UPSERT": "1"},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"process_queue failed: {proc.stderr[:500]}")

    if staging_items:
        set_staging_status_for_ids(
            settings.queue_db_path,
            [str(item.get("id")) for item in staging_items],
            "discarded",
        )

    if settings.profile_enable:
        extract = _run_profile_extract(
            settings,
            conversation_text=conversation_text,
            staging_items=staging_items,
        )
        static_items = extract.get("static", [])
        dynamic_items = extract.get("dynamic", [])
        if not isinstance(static_items, list):
            static_items = []
        if not isinstance(dynamic_items, list):
            dynamic_items = []

        promoted_profile = 0
        for item in static_items[: settings.profile_max_items]:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("fact_text", "")).strip()
            fact_type = str(item.get("fact_type", "fact")).strip().lower()
            if not fact_text or fact_type not in {"decision", "preference", "constraint", "fact"}:
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5) or 0.5)))
            _upsert_memory_item(
                settings,
                scope="global_profile",
                fact_text=fact_text,
                fact_type=fact_type,
                confidence=confidence,
                source="session_stop",
                evidence_increment=max(1, settings.profile_min_evidence),
                actor="session_stop",
            )
            promoted_profile += 1

        for item in dynamic_items[: settings.profile_max_items]:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("fact_text", "")).strip()
            fact_type = str(item.get("fact_type", "fact")).strip().lower()
            if not fact_text or fact_type not in {"decision", "preference", "constraint", "fact"}:
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.4) or 0.4)))
            _upsert_memory_item(
                settings,
                scope="working_memory",
                fact_text=fact_text,
                fact_type=fact_type,
                confidence=confidence,
                source="session_stop",
                evidence_increment=1,
                actor="session_stop",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        window_start = (now - timedelta(hours=settings.profile_dynamic_hours)).isoformat()
        _compact_profile_snapshot(
            settings,
            created_by="session_stop",
            source_window_start=window_start,
            source_window_end=now.isoformat(),
        )
        log(
            "session_stop profile extracted",
            event_id=event_id,
            session_id=session_id,
            static_candidates=len(static_items),
            dynamic_candidates=len(dynamic_items),
            promoted_profile=promoted_profile,
            error=extract.get("error", ""),
        )

    delete_session_buffer(settings.queue_db_path, session_id)
    cleanup_expired_staging(settings.queue_db_path)

    log("session_stop done", event_id=event_id, session_id=session_id, turn_count=turn_count)


def _handle_turn(settings, payload: dict, session_id: str, agent: str, event_id: str) -> None:
    turn_index = int(payload.get("turn_index") or 0)
    role = str(payload.get("role") or "").strip().lower()
    text = str(payload.get("text") or "").strip()
    cwd = str(payload.get("cwd") or "")
    ts = str(payload.get("ts") or "")

    if turn_index < 1:
        raise ValueError("turn missing turn_index")
    if role not in {"user", "assistant"}:
        raise ValueError("turn missing role")
    if not text:
        raise ValueError("turn missing text")

    cleanup_expired_staging(settings.queue_db_path)

    recent_turns = upsert_session_buffer(
        settings.queue_db_path,
        session_id=session_id,
        agent=agent,
        turn_index=turn_index,
        role=role,
        text=text[:4000],
        cwd=cwd,
        ts=ts,
        window_size=settings.staging_recent_turns_window,
    )

    promoted_ids: set[str] = set()

    if turn_index % settings.turn_live_cadence != 0:
        if role == "user" and _has_explicit_confirmation(text):
            experimental = list_staging_for_session(
                settings.queue_db_path,
                session_id=session_id,
                statuses=("experimental",),
                limit=30,
            )
            promoted_ids.update(str(item.get("id")) for item in experimental if item.get("id"))
        if promoted_ids:
            set_staging_status_for_ids(settings.queue_db_path, list(promoted_ids), "confirmed")
            log("turn promote confirmed", event_id=event_id, promoted=len(promoted_ids))
        log("turn buffered", event_id=event_id, turn_index=turn_index)
        return

    counts = get_stats(settings.queue_db_path)["counts"]
    if (counts["queued"] + counts["processing"]) >= settings.backpressure_queue_threshold:
        log(
            "turn skipped backpressure",
            event_id=event_id,
            turn_index=turn_index,
            queued=counts["queued"],
            processing=counts["processing"],
        )
        return

    result = _run_live_extract(settings, session_id=session_id, agent=agent, turns=recent_turns)
    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    touched_ids: list[str] = []
    for candidate in candidates[: settings.live_extract_max_candidates]:
        if not isinstance(candidate, dict):
            continue
        fact_text = str(candidate.get("fact_text") or "").strip()
        if not fact_text:
            continue
        fact_type = str(candidate.get("fact_type") or "fact").strip().lower()
        if fact_type not in {"decision", "preference", "constraint", "fact"}:
            fact_type = "fact"
        try:
            confidence = float(candidate.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        evidence_excerpt = str(candidate.get("evidence_excerpt") or fact_text).strip()

        upserted = upsert_staging_memory(
            settings.queue_db_path,
            session_id=session_id,
            agent=agent,
            fingerprint=_fingerprint(fact_text, fact_type),
            fact_text=fact_text,
            fact_type=fact_type,
            confidence=confidence,
            source_last_turn=turn_index,
            ttl_hours=settings.staging_ttl_hours,
        )
        staging_id = str(upserted["id"])
        touched_ids.append(staging_id)
        add_staging_evidence(
            settings.queue_db_path,
            staging_id=staging_id,
            turn_index=turn_index,
            excerpt=evidence_excerpt,
        )

        if int(upserted.get("evidence_count", 1)) >= settings.promotion_min_evidence:
            promoted_ids.add(staging_id)

    if role == "user" and _has_explicit_confirmation(text):
        if touched_ids:
            promoted_ids.update(touched_ids)
        else:
            experimental = list_staging_for_session(
                settings.queue_db_path,
                session_id=session_id,
                statuses=("experimental",),
                limit=30,
            )
            promoted_ids.update(str(item.get("id")) for item in experimental if item.get("id"))

    if promoted_ids:
        set_staging_status_for_ids(settings.queue_db_path, list(promoted_ids), "confirmed")
        if settings.profile_enable:
            confirmed = list_staging_for_session(
                settings.queue_db_path,
                session_id=session_id,
                statuses=("confirmed",),
                limit=50,
            )
            promoted_set = set(promoted_ids)
            for item in confirmed:
                sid = str(item.get("id", ""))
                if sid not in promoted_set:
                    continue
                fact_text = str(item.get("fact_text", "")).strip()
                fact_type = str(item.get("fact_type", "fact")).strip().lower()
                if not fact_text or fact_type not in {"decision", "preference", "constraint", "fact"}:
                    continue
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.4) or 0.4)))
                _upsert_memory_item(
                    settings,
                    scope="working_memory",
                    fact_text=fact_text,
                    fact_type=fact_type,
                    confidence=confidence,
                    source="turn_live",
                    evidence_increment=max(1, int(item.get("evidence_count", 1) or 1)),
                    actor=agent,
                )

    log(
        "turn live processed",
        event_id=event_id,
        session_id=session_id,
        agent=agent,
        turn_index=turn_index,
        candidates=len(touched_ids),
        promoted=len(promoted_ids),
        error=result.get("error", ""),
    )


def _handle_memory_forget(settings, payload: dict, event_id: str) -> None:
    if not settings.profile_enable:
        raise ValueError("profile layer disabled")
    if not settings.forget_soft_delete:
        raise ValueError("soft forget disabled")
    memory_id = str(payload.get("memory_id", "")).strip()
    reason = str(payload.get("reason", "manual_admin_forget")).strip() or "manual_admin_forget"
    actor = str(payload.get("actor", "admin")).strip() or "admin"
    if not memory_id:
        raise ValueError("memory_forget missing memory_id")
    result = forget_memory_node(settings.queue_db_path, memory_id=memory_id, reason=reason)
    append_memory_audit(
        settings.queue_db_path,
        action="forget",
        target_id=memory_id,
        payload={"reason": reason, "result": result},
        actor=actor,
    )
    log(
        "memory_action",
        action="forget",
        event_id=event_id,
        memory_id=memory_id,
        global_key=result.get("global_key"),
        from_version=result.get("version"),
        to_version=result.get("version"),
    )


def _handle_memory_restore(settings, payload: dict, event_id: str) -> None:
    if not settings.profile_enable:
        raise ValueError("profile layer disabled")
    memory_id = str(payload.get("memory_id", "")).strip()
    actor = str(payload.get("actor", "admin")).strip() or "admin"
    if not memory_id:
        raise ValueError("memory_restore missing memory_id")
    result = restore_memory_node(settings.queue_db_path, memory_id=memory_id)
    append_memory_audit(
        settings.queue_db_path,
        action="restore",
        target_id=memory_id,
        payload={"result": result},
        actor=actor,
    )
    log(
        "memory_action",
        action="restore",
        event_id=event_id,
        memory_id=memory_id,
        global_key=result.get("global_key"),
        from_version=result.get("version"),
        to_version=result.get("version"),
    )


def _handle_memory_upsert(settings, payload: dict, event_id: str) -> None:
    if not settings.profile_enable:
        raise ValueError("profile layer disabled")
    fact_text = str(payload.get("fact_text", "")).strip()
    fact_type = str(payload.get("fact_type", "fact")).strip().lower()
    scope = str(payload.get("scope", "global_profile")).strip().lower()
    actor = str(payload.get("actor", "admin")).strip() or "admin"
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5) or 0.5)))
    if not fact_text:
        raise ValueError("memory_upsert missing fact_text")
    if fact_type not in {"decision", "preference", "constraint", "fact"}:
        raise ValueError("memory_upsert invalid fact_type")
    if scope not in {"global_profile", "working_memory"}:
        raise ValueError("memory_upsert invalid scope")

    result = _upsert_memory_item(
        settings,
        scope=scope,
        fact_text=fact_text,
        fact_type=fact_type,
        confidence=confidence,
        source="admin",
        evidence_increment=max(1, settings.profile_min_evidence),
        actor=actor,
    )
    log(
        "memory_action",
        action="admin_upsert",
        event_id=event_id,
        global_key=result.get("global_key"),
        from_version=result.get("from_version"),
        to_version=result.get("to_version"),
    )


def _handle_profile_compact(settings, payload: dict, event_id: str) -> None:
    if not settings.profile_enable:
        raise ValueError("profile layer disabled")
    actor = str(payload.get("actor", "hourly_compact")).strip() or "hourly_compact"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window_start = (now - timedelta(hours=settings.profile_dynamic_hours)).isoformat()
    snapshot = _compact_profile_snapshot(
        settings,
        created_by="hourly_compact" if actor == "hourly_compact_job" else actor,
        source_window_start=window_start,
        source_window_end=now.isoformat(),
    )
    log(
        "memory_action",
        action="profile_compact",
        event_id=event_id,
        global_key="profile:global",
        from_version=int(snapshot.get("version", 1)) - 1,
        to_version=snapshot.get("version"),
    )


def _handle_relation_compact(settings, payload: dict, event_id: str) -> None:
    if not settings.relation_enable:
        log("relation_compact skipped", event_id=event_id, reason="disabled")
        return

    actor = str(payload.get("actor", "relation_compact_job")).strip() or "relation_compact_job"
    nodes = list_all_active_memory_nodes(
        settings.queue_db_path,
        limit=max(50, settings.relation_batch_max_pairs * 3),
    )
    generated = generate_relation_candidates(nodes, settings)
    accepted = generated.get("accepted", [])
    rejected = generated.get("rejected", [])
    llm_error = str(generated.get("llm_error", "") or "")
    stats = generated.get("stats", {})

    for row in rejected:
        upsert_relation_candidate(
            settings.queue_db_path,
            src_node_id=str(row.get("src_node_id", "")),
            dst_node_id=str(row.get("dst_node_id", "")),
            relation=str(row.get("relation", "")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            decision_source=str(row.get("decision_source", "deterministic")),
            status="rejected",
            reason=str(row.get("reason", "")),
            canonical_key=str(row.get("canonical_key", "")),
        )

    inserted = 0
    considered = 0
    for row in accepted:
        considered += 1
        canonical_key = str(row.get("canonical_key", ""))
        upsert_relation_candidate(
            settings.queue_db_path,
            src_node_id=str(row.get("src_node_id", "")),
            dst_node_id=str(row.get("dst_node_id", "")),
            relation=str(row.get("relation", "")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            decision_source=str(row.get("decision_source", "deterministic")),
            status="accepted",
            reason=str(row.get("reason", "")),
            canonical_key=canonical_key,
        )
        if not settings.relation_write:
            continue
        if inserted >= settings.relation_max_new_edges_per_run:
            break
        created, edge_id = insert_memory_edge_if_missing(
            settings.queue_db_path,
            src_node_id=str(row.get("src_node_id", "")),
            dst_node_id=str(row.get("dst_node_id", "")),
            relation=str(row.get("relation", "")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
        )
        if not created:
            continue
        inserted += 1
        append_memory_audit(
            settings.queue_db_path,
            action="relation_link",
            target_id=edge_id,
            payload={
                "canonical_key": canonical_key,
                "relation": row.get("relation"),
                "source": row.get("decision_source"),
                "confidence": row.get("confidence"),
                "event_id": event_id,
            },
            actor=actor,
        )

    log(
        "memory_action",
        action="relation_compact",
        event_id=event_id,
        relation_write=bool(settings.relation_write),
        considered=considered,
        inserted=inserted,
        rejected=len(rejected),
        llm_error=llm_error,
        stats=stats,
    )


def process_event(settings, event: dict) -> None:
    event_type = event["event_type"]
    payload = event["payload"]

    if event_type == "note_add":
        _handle_note_add(settings, payload, event["id"])
        return
    if event_type == "session_stop":
        _handle_session_stop(settings, payload, event["session_id"], event["id"])
        return
    if event_type == "turn":
        _handle_turn(settings, payload, event["session_id"], event["agent"], event["id"])
        return
    if event_type == "memory_forget":
        _handle_memory_forget(settings, payload, event["id"])
        return
    if event_type == "memory_restore":
        _handle_memory_restore(settings, payload, event["id"])
        return
    if event_type == "memory_upsert":
        _handle_memory_upsert(settings, payload, event["id"])
        return
    if event_type == "profile_compact":
        _handle_profile_compact(settings, payload, event["id"])
        return
    if event_type == "relation_compact":
        _handle_relation_compact(settings, payload, event["id"])
        return

    raise ValueError(f"Unsupported event_type: {event_type}")


def run_worker() -> int:
    settings = load_settings()
    init_db(settings.queue_db_path)

    lock = WorkerLock(settings.worker_lock_path)
    try:
        lock.acquire()
    except BlockingIOError:
        log("Another worker is already running", lock_path=str(settings.worker_lock_path))
        return 1

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        log("stop requested", signal=signum)
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log("Worker started")

    try:
        while not stop:
            event = claim_next_event(settings.queue_db_path)
            if event is None:
                time.sleep(settings.poll_interval_seconds)
                continue

            event_id = event["id"]
            log(
                "Processing event",
                event_id=event_id,
                event_type=event["event_type"],
                agent=event["agent"],
                session_id=event["session_id"],
            )
            try:
                process_event(settings, event)
                mark_done(settings.queue_db_path, event_id)
                log("Done", event_id=event_id)
            except Exception as exc:
                mark_error(settings.queue_db_path, event_id, str(exc))
                log("Error", event_id=event_id, error=str(exc))

    finally:
        lock.release()
        log("Worker stopped")

    return 0


def main() -> None:
    sys.exit(run_worker())


if __name__ == "__main__":
    main()

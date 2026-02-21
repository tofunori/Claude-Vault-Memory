from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                agent TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                dedup_hash TEXT NOT NULL UNIQUE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_status_created ON events(status, created_at);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_buffers (
                session_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                last_turn_index INTEGER NOT NULL,
                recent_turns_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_memories (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                status TEXT NOT NULL,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                confidence REAL NOT NULL DEFAULT 0.0,
                source_last_turn INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                UNIQUE(session_id, fingerprint)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_evidence (
                id TEXT PRIMARY KEY,
                staging_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                excerpt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(staging_id) REFERENCES staging_memories(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_staging_session_status_expires
            ON staging_memories(session_id, status, expires_at);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_staging_status_expires
            ON staging_memories(status, expires_at);
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                global_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                scope TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                supersedes_id TEXT,
                forget_reason TEXT,
                UNIQUE(global_key, version)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_nodes_global_key ON memory_nodes(global_key);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_nodes_status_scope ON memory_nodes(status, scope);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_edges (
                id TEXT PRIMARY KEY,
                src_node_id TEXT NOT NULL,
                dst_node_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_edges_src ON memory_edges(src_node_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_edges_dst ON memory_edges(dst_node_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_edges_relation_src_dst ON memory_edges(relation, src_node_id, dst_node_id);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS note_aliases (
                alias TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'admin',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_aliases_note_id ON note_aliases(note_id);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_candidates (
                id TEXT PRIMARY KEY,
                src_node_id TEXT NOT NULL,
                dst_node_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                decision_source TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                canonical_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_candidates_status ON relation_candidates(status, updated_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_candidates_source ON relation_candidates(decision_source, updated_at);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_snapshots (
                id TEXT PRIMARY KEY,
                profile_name TEXT NOT NULL DEFAULT 'global',
                version INTEGER NOT NULL,
                static_json TEXT NOT NULL,
                dynamic_json TEXT NOT NULL,
                source_window_start TEXT,
                source_window_end TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                UNIQUE(profile_name, version)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_audit (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                target_id TEXT,
                payload_json TEXT NOT NULL,
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_created ON memory_audit(created_at);"
        )
        conn.commit()


def enqueue_event(
    db_path: Path,
    *,
    event_id: str,
    event_type: str,
    agent: str,
    session_id: str,
    payload: dict[str, Any],
    dedup_hash: str,
) -> tuple[bool, str]:
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    now = utc_now_iso()

    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO events(id, event_type, agent, session_id, payload_json, status, created_at, dedup_hash)
                VALUES(?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (event_id, event_type, agent, session_id, payload_json, now, dedup_hash),
            )
            conn.commit()
            return True, event_id
        except sqlite3.IntegrityError:
            row = conn.execute(
                "SELECT id FROM events WHERE dedup_hash = ? LIMIT 1", (dedup_hash,)
            ).fetchone()
            conn.commit()
            return False, (row["id"] if row else event_id)


def claim_next_event(db_path: Path) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, event_type, agent, session_id, payload_json, created_at
            FROM events
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        now = utc_now_iso()
        updated = conn.execute(
            """
            UPDATE events
            SET status='processing', started_at=?
            WHERE id=? AND status='queued'
            """,
            (now, row["id"]),
        ).rowcount

        if updated != 1:
            conn.commit()
            return None

        conn.commit()
        return {
            "id": row["id"],
            "event_type": row["event_type"],
            "agent": row["agent"],
            "session_id": row["session_id"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }


def mark_done(db_path: Path, event_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE events SET status='done', finished_at=?, error=NULL WHERE id=?",
            (utc_now_iso(), event_id),
        )
        conn.commit()


def mark_error(db_path: Path, event_id: str, error: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE events SET status='error', finished_at=?, error=? WHERE id=?",
            (utc_now_iso(), error[:2000], event_id),
        )
        conn.commit()


def get_stats(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as conn:
        totals = {
            row["status"]: row["count"]
            for row in conn.execute(
                "SELECT status, COUNT(*) as count FROM events GROUP BY status"
            )
        }
        queued_oldest = conn.execute(
            "SELECT created_at FROM events WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()

    return {
        "counts": {
            "queued": totals.get("queued", 0),
            "processing": totals.get("processing", 0),
            "done": totals.get("done", 0),
            "error": totals.get("error", 0),
        },
        "oldest_queued_at": queued_oldest["created_at"] if queued_oldest else None,
    }


def upsert_session_buffer(
    db_path: Path,
    *,
    session_id: str,
    agent: str,
    turn_index: int,
    role: str,
    text: str,
    cwd: str | None,
    ts: str | None,
    window_size: int = 12,
) -> list[dict[str, Any]]:
    now = utc_now_iso()
    turn = {
        "turn_index": turn_index,
        "role": role,
        "text": text,
        "cwd": cwd or "",
        "ts": ts or now,
    }

    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT recent_turns_json, last_turn_index FROM session_buffers WHERE session_id=?",
            (session_id,),
        ).fetchone()

        recent: list[dict[str, Any]] = []
        if row and row["recent_turns_json"]:
            try:
                data = json.loads(row["recent_turns_json"])
                if isinstance(data, list):
                    recent = [x for x in data if isinstance(x, dict)]
            except Exception:
                recent = []

        replaced = False
        for i, existing in enumerate(recent):
            if int(existing.get("turn_index", -1)) == turn_index:
                recent[i] = turn
                replaced = True
                break
        if not replaced:
            recent.append(turn)

        recent.sort(key=lambda x: int(x.get("turn_index", 0)))
        if len(recent) > window_size:
            recent = recent[-window_size:]

        recent_json = json.dumps(recent, ensure_ascii=False, separators=(",", ":"))
        conn.execute(
            """
            INSERT INTO session_buffers(session_id, agent, last_turn_index, recent_turns_json, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                agent=excluded.agent,
                last_turn_index=excluded.last_turn_index,
                recent_turns_json=excluded.recent_turns_json,
                updated_at=excluded.updated_at
            """,
            (session_id, agent, turn_index, recent_json, now),
        )
        conn.commit()
    return recent


def delete_session_buffer(db_path: Path, session_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM session_buffers WHERE session_id=?", (session_id,))
        conn.commit()


def cleanup_expired_staging(db_path: Path) -> int:
    now = utc_now_iso()
    with connect(db_path) as conn:
        deleted = conn.execute(
            """
            DELETE FROM staging_memories
            WHERE expires_at <= ? AND status IN ('experimental', 'discarded')
            """,
            (now,),
        ).rowcount
        conn.commit()
    return int(deleted or 0)


def add_staging_evidence(
    db_path: Path,
    *,
    staging_id: str,
    turn_index: int,
    excerpt: str,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO staging_evidence(id, staging_id, turn_index, excerpt, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (str(uuid4()), staging_id, turn_index, excerpt[:1000], utc_now_iso()),
        )
        conn.commit()


def upsert_staging_memory(
    db_path: Path,
    *,
    session_id: str,
    agent: str,
    fingerprint: str,
    fact_text: str,
    fact_type: str,
    confidence: float,
    source_last_turn: int,
    ttl_hours: int,
) -> dict[str, Any]:
    now = utc_now_iso()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).replace(
        microsecond=0
    ).isoformat()

    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, evidence_count, confidence, status
            FROM staging_memories
            WHERE session_id=? AND fingerprint=?
            LIMIT 1
            """,
            (session_id, fingerprint),
        ).fetchone()

        if row:
            new_count = int(row["evidence_count"]) + 1
            new_confidence = max(float(row["confidence"] or 0.0), float(confidence))
            current_status = row["status"]
            next_status = "experimental" if current_status == "discarded" else current_status
            conn.execute(
                """
                UPDATE staging_memories
                SET fact_text=?,
                    fact_type=?,
                    confidence=?,
                    evidence_count=?,
                    source_last_turn=?,
                    status=?,
                    updated_at=?,
                    expires_at=?
                WHERE id=?
                """,
                (
                    fact_text[:600],
                    fact_type[:40],
                    new_confidence,
                    new_count,
                    source_last_turn,
                    next_status,
                    now,
                    expires_at,
                    row["id"],
                ),
            )
            conn.commit()
            return {
                "id": row["id"],
                "evidence_count": new_count,
                "confidence": new_confidence,
                "status": next_status,
            }

        staging_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO staging_memories(
                id, session_id, agent, fingerprint, fact_text, fact_type, status,
                evidence_count, confidence, source_last_turn, created_at, updated_at, expires_at
            )
            VALUES(?, ?, ?, ?, ?, ?, 'experimental', 1, ?, ?, ?, ?, ?)
            """,
            (
                staging_id,
                session_id,
                agent,
                fingerprint,
                fact_text[:600],
                fact_type[:40],
                float(confidence),
                source_last_turn,
                now,
                now,
                expires_at,
            ),
        )
        conn.commit()
        return {
            "id": staging_id,
            "evidence_count": 1,
            "confidence": float(confidence),
            "status": "experimental",
        }


def set_staging_status_for_ids(db_path: Path, ids: list[str], status: str) -> int:
    ids = [x for x in ids if x]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    params = [status, utc_now_iso(), *ids]
    with connect(db_path) as conn:
        changed = conn.execute(
            f"UPDATE staging_memories SET status=?, updated_at=? WHERE id IN ({placeholders})",
            params,
        ).rowcount
        conn.commit()
    return int(changed or 0)


def set_staging_status_for_session(db_path: Path, session_id: str, status: str) -> int:
    with connect(db_path) as conn:
        changed = conn.execute(
            """
            UPDATE staging_memories
            SET status=?, updated_at=?
            WHERE session_id=? AND status != ?
            """,
            (status, utc_now_iso(), session_id, status),
        ).rowcount
        conn.commit()
    return int(changed or 0)


def list_staging_for_session(
    db_path: Path,
    *,
    session_id: str,
    statuses: tuple[str, ...] = ("experimental",),
    limit: int = 10,
    only_unexpired: bool = True,
) -> list[dict[str, Any]]:
    if not statuses:
        return []
    placeholders = ",".join("?" for _ in statuses)
    where = [f"session_id=? AND status IN ({placeholders})"]
    params: list[Any] = [session_id, *statuses]
    if only_unexpired:
        where.append("expires_at > ?")
        params.append(utc_now_iso())
    where_sql = " AND ".join(where)

    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, session_id, agent, fact_text, fact_type, status, evidence_count, confidence, source_last_turn, created_at, updated_at, expires_at
            FROM staging_memories
            WHERE {where_sql}
            ORDER BY confidence DESC, evidence_count DESC, updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({k: row[k] for k in row.keys()})
    return out


def append_memory_audit(
    db_path: Path,
    *,
    action: str,
    target_id: str | None,
    payload: dict[str, Any],
    actor: str,
) -> str:
    audit_id = str(uuid4())
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO memory_audit(id, action, target_id, payload_json, actor, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                action[:40],
                target_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                actor[:80],
                utc_now_iso(),
            ),
        )
        conn.commit()
    return audit_id


def insert_memory_edge(
    db_path: Path,
    *,
    src_node_id: str,
    dst_node_id: str,
    relation: str,
    confidence: float,
) -> str:
    edge_id = str(uuid4())
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO memory_edges(id, src_node_id, dst_node_id, relation, confidence, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                src_node_id,
                dst_node_id,
                relation[:32],
                max(0.0, min(1.0, float(confidence))),
                utc_now_iso(),
            ),
        )
        conn.commit()
    return edge_id


def canonical_memory_edge_key(src_node_id: str, dst_node_id: str, relation: str) -> str:
    rel = relation.strip().lower()[:32]
    src = src_node_id.strip()
    dst = dst_node_id.strip()
    if rel in {"supports", "contradicts"}:
        a, b = sorted([src, dst])
        return f"{rel}|{a}|{b}"
    return f"{rel}|{src}|{dst}"


def memory_edge_exists(
    db_path: Path,
    *,
    src_node_id: str,
    dst_node_id: str,
    relation: str,
) -> bool:
    rel = relation.strip().lower()[:32]
    src = src_node_id.strip()
    dst = dst_node_id.strip()
    with connect(db_path) as conn:
        if rel in {"supports", "contradicts"}:
            row = conn.execute(
                """
                SELECT id
                FROM memory_edges
                WHERE relation=?
                  AND ((src_node_id=? AND dst_node_id=?) OR (src_node_id=? AND dst_node_id=?))
                LIMIT 1
                """,
                (rel, src, dst, dst, src),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id
                FROM memory_edges
                WHERE relation=? AND src_node_id=? AND dst_node_id=?
                LIMIT 1
                """,
                (rel, src, dst),
            ).fetchone()
    return row is not None


def insert_memory_edge_if_missing(
    db_path: Path,
    *,
    src_node_id: str,
    dst_node_id: str,
    relation: str,
    confidence: float,
) -> tuple[bool, str]:
    rel = relation.strip().lower()[:32]
    src = src_node_id.strip()
    dst = dst_node_id.strip()
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if rel in {"supports", "contradicts"}:
            existing = conn.execute(
                """
                SELECT id
                FROM memory_edges
                WHERE relation=?
                  AND ((src_node_id=? AND dst_node_id=?) OR (src_node_id=? AND dst_node_id=?))
                LIMIT 1
                """,
                (rel, src, dst, dst, src),
            ).fetchone()
        else:
            existing = conn.execute(
                """
                SELECT id
                FROM memory_edges
                WHERE relation=? AND src_node_id=? AND dst_node_id=?
                LIMIT 1
                """,
                (rel, src, dst),
            ).fetchone()
        if existing is not None:
            conn.commit()
            return False, str(existing["id"])
        edge_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO memory_edges(id, src_node_id, dst_node_id, relation, confidence, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                src,
                dst,
                rel,
                max(0.0, min(1.0, float(confidence))),
                now,
            ),
        )
        conn.commit()
    return True, edge_id


def get_memory_node_by_id(db_path: Path, node_id: str) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence,
                   evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            FROM memory_nodes
            WHERE id=?
            LIMIT 1
            """,
            (node_id,),
        ).fetchone()
    if not row:
        return None
    return {k: row[k] for k in row.keys()}


def get_active_memory_node_by_key(db_path: Path, global_key: str) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence,
                   evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            FROM memory_nodes
            WHERE global_key=? AND status='active'
            ORDER BY version DESC
            LIMIT 1
            """,
            (global_key,),
        ).fetchone()
    if not row:
        return None
    return {k: row[k] for k in row.keys()}


def upsert_memory_node_versioned(
    db_path: Path,
    *,
    global_key: str,
    scope: str,
    fact_text: str,
    fact_type: str,
    confidence: float,
    source: str,
    relation_mode: str,
    evidence_increment: int = 1,
) -> dict[str, Any]:
    now = utc_now_iso()
    relation_mode = relation_mode if relation_mode in {"same", "updates", "contradicts"} else "updates"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            """
            SELECT id, version, fact_text, fact_type, scope, confidence, evidence_count
            FROM memory_nodes
            WHERE global_key=? AND status='active'
            ORDER BY version DESC
            LIMIT 1
            """,
            (global_key,),
        ).fetchone()

        if current is None:
            node_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO memory_nodes(
                    id, global_key, version, scope, fact_text, fact_type, status, confidence,
                    evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
                )
                VALUES(?, ?, 1, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    node_id,
                    global_key,
                    scope[:40],
                    fact_text[:600],
                    fact_type[:40],
                    max(0.0, min(1.0, float(confidence))),
                    max(1, int(evidence_increment)),
                    source[:40],
                    now,
                    now,
                ),
            )
            conn.commit()
            return {
                "action": "created",
                "node_id": node_id,
                "global_key": global_key,
                "version": 1,
                "from_version": None,
                "to_version": 1,
                "edge_relation": None,
            }

        current_id = str(current["id"])
        current_version = int(current["version"])
        if relation_mode == "same":
            next_conf = max(float(current["confidence"] or 0.0), max(0.0, min(1.0, float(confidence))))
            next_evidence = int(current["evidence_count"] or 0) + max(1, int(evidence_increment))
            conn.execute(
                """
                UPDATE memory_nodes
                SET confidence=?, evidence_count=?, updated_at=?, fact_text=?, fact_type=?, scope=?
                WHERE id=?
                """,
                (
                    next_conf,
                    next_evidence,
                    now,
                    fact_text[:600],
                    fact_type[:40],
                    scope[:40],
                    current_id,
                ),
            )
            conn.commit()
            return {
                "action": "updated",
                "node_id": current_id,
                "global_key": global_key,
                "version": current_version,
                "from_version": current_version,
                "to_version": current_version,
                "edge_relation": None,
            }

        conn.execute(
            "UPDATE memory_nodes SET status='superseded', updated_at=? WHERE id=?",
            (now, current_id),
        )
        new_id = str(uuid4())
        new_version = current_version + 1
        conn.execute(
            """
            INSERT INTO memory_nodes(
                id, global_key, version, scope, fact_text, fact_type, status, confidence,
                evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            )
            VALUES(?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                new_id,
                global_key,
                new_version,
                scope[:40],
                fact_text[:600],
                fact_type[:40],
                max(0.0, min(1.0, float(confidence))),
                max(1, int(evidence_increment)),
                source[:40],
                now,
                now,
                current_id,
            ),
        )
        edge_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO memory_edges(id, src_node_id, dst_node_id, relation, confidence, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                new_id,
                current_id,
                relation_mode,
                max(0.0, min(1.0, float(confidence))),
                now,
            ),
        )
        conn.commit()
        return {
            "action": "versioned",
            "node_id": new_id,
            "global_key": global_key,
            "version": new_version,
            "from_version": current_version,
            "to_version": new_version,
            "edge_relation": relation_mode,
            "superseded_id": current_id,
            "edge_id": edge_id,
        }


def forget_memory_node(
    db_path: Path,
    *,
    memory_id: str,
    reason: str,
) -> dict[str, Any]:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, global_key, version, status
            FROM memory_nodes
            WHERE id=?
            LIMIT 1
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            conn.commit()
            raise ValueError("memory_id not found")
        if row["status"] == "forgotten":
            conn.commit()
            return {
                "action": "noop",
                "node_id": memory_id,
                "global_key": row["global_key"],
                "version": int(row["version"]),
            }
        conn.execute(
            """
            UPDATE memory_nodes
            SET status='forgotten', forget_reason=?, updated_at=?
            WHERE id=?
            """,
            (reason[:500], now, memory_id),
        )
        conn.commit()
        return {
            "action": "forgotten",
            "node_id": memory_id,
            "global_key": row["global_key"],
            "version": int(row["version"]),
        }


def restore_memory_node(db_path: Path, *, memory_id: str) -> dict[str, Any]:
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, global_key, version, status
            FROM memory_nodes
            WHERE id=?
            LIMIT 1
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            conn.commit()
            raise ValueError("memory_id not found")
        if row["status"] != "forgotten":
            conn.commit()
            return {
                "action": "noop",
                "node_id": memory_id,
                "global_key": row["global_key"],
                "version": int(row["version"]),
            }
        conflict = conn.execute(
            """
            SELECT id FROM memory_nodes
            WHERE global_key=? AND status='active' AND id<>?
            LIMIT 1
            """,
            (row["global_key"], memory_id),
        ).fetchone()
        if conflict is not None:
            conn.commit()
            raise ValueError("restore conflict: active version already exists")
        conn.execute(
            """
            UPDATE memory_nodes
            SET status='active', updated_at=?, forget_reason=NULL
            WHERE id=?
            """,
            (now, memory_id),
        )
        conn.commit()
        return {
            "action": "restored",
            "node_id": memory_id,
            "global_key": row["global_key"],
            "version": int(row["version"]),
        }


def list_active_memory_nodes(
    db_path: Path,
    *,
    scope: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence,
                   evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            FROM memory_nodes
            WHERE scope=? AND status='active'
            ORDER BY confidence DESC, evidence_count DESC, updated_at DESC
            LIMIT ?
            """,
            (scope, max(1, limit)),
        ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


def list_recent_active_memory_nodes(
    db_path: Path,
    *,
    scope: str,
    updated_after: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence,
                   evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            FROM memory_nodes
            WHERE scope=? AND status='active' AND updated_at>=?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (scope, updated_after, max(1, limit)),
        ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


def list_all_active_memory_nodes(
    db_path: Path,
    *,
    limit: int = 1200,
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, global_key, version, scope, fact_text, fact_type, status, confidence,
                   evidence_count, source, created_at, updated_at, supersedes_id, forget_reason
            FROM memory_nodes
            WHERE status='active'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


def upsert_note_alias(
    db_path: Path,
    *,
    alias: str,
    note_id: str,
    source: str = "admin",
) -> bool:
    normalized_alias = alias.strip().lower()
    normalized_note = note_id.strip()
    if not normalized_alias or not normalized_note:
        return False
    now = utc_now_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO note_aliases(alias, note_id, source, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(alias) DO UPDATE SET
                note_id=excluded.note_id,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            (normalized_alias[:120], normalized_note[:240], source[:40], now, now),
        )
        conn.commit()
    return True


def list_note_aliases(db_path: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT alias, note_id, source, created_at, updated_at
            FROM note_aliases
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
    return [{k: row[k] for k in row.keys()} for row in rows]


def upsert_relation_candidate(
    db_path: Path,
    *,
    src_node_id: str,
    dst_node_id: str,
    relation: str,
    confidence: float,
    decision_source: str,
    status: str,
    reason: str,
    canonical_key: str,
) -> tuple[bool, str]:
    now = utc_now_iso()
    candidate_id = str(uuid4())
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT id FROM relation_candidates WHERE canonical_key=? LIMIT 1",
            (canonical_key,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO relation_candidates(
                    id, src_node_id, dst_node_id, relation, confidence, decision_source, status, reason,
                    canonical_key, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    src_node_id,
                    dst_node_id,
                    relation[:32],
                    max(0.0, min(1.0, float(confidence))),
                    decision_source[:40],
                    status[:24],
                    reason[:500],
                    canonical_key[:200],
                    now,
                    now,
                ),
            )
            conn.commit()
            return True, candidate_id
        conn.execute(
            """
            UPDATE relation_candidates
            SET src_node_id=?,
                dst_node_id=?,
                relation=?,
                confidence=?,
                decision_source=?,
                status=?,
                reason=?,
                updated_at=?
            WHERE canonical_key=?
            """,
            (
                src_node_id,
                dst_node_id,
                relation[:32],
                max(0.0, min(1.0, float(confidence))),
                decision_source[:40],
                status[:24],
                reason[:500],
                now,
                canonical_key[:200],
            ),
        )
        row = conn.execute(
            "SELECT id FROM relation_candidates WHERE canonical_key=? LIMIT 1",
            (canonical_key[:200],),
        ).fetchone()
        conn.commit()
    return False, str(row["id"]) if row else candidate_id


def get_relation_stats(db_path: Path) -> dict[str, Any]:
    with connect(db_path) as conn:
        by_status = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM relation_candidates
            GROUP BY status
            """
        ).fetchall()
        by_source = conn.execute(
            """
            SELECT decision_source, COUNT(*) AS count
            FROM relation_candidates
            GROUP BY decision_source
            """
        ).fetchall()
        edge_count_row = conn.execute(
            "SELECT COUNT(*) AS count FROM memory_edges"
        ).fetchone()
        active_nodes_row = conn.execute(
            "SELECT COUNT(*) AS count FROM memory_nodes WHERE status='active'"
        ).fetchone()
        singleton_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_nodes n
            WHERE n.status='active'
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_edges e
                  WHERE e.src_node_id=n.id OR e.dst_node_id=n.id
              )
            """
        ).fetchone()
        recent_row = conn.execute(
            """
            SELECT MAX(updated_at) AS last_candidate_at
            FROM relation_candidates
            """
        ).fetchone()

    return {
        "candidate_counts": {str(row["status"]): int(row["count"]) for row in by_status},
        "source_counts": {str(row["decision_source"]): int(row["count"]) for row in by_source},
        "memory_edges_total": int(edge_count_row["count"]) if edge_count_row else 0,
        "memory_nodes_active": int(active_nodes_row["count"]) if active_nodes_row else 0,
        "memory_singletons": int(singleton_row["count"]) if singleton_row else 0,
        "last_candidate_at": recent_row["last_candidate_at"] if recent_row else None,
    }


def create_profile_snapshot(
    db_path: Path,
    *,
    profile_name: str,
    static_items: list[dict[str, Any]],
    dynamic_items: list[dict[str, Any]],
    created_by: str,
    source_window_start: str | None = None,
    source_window_end: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    pname = (profile_name or "global")[:64]
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT version
            FROM profile_snapshots
            WHERE profile_name=?
            ORDER BY version DESC
            LIMIT 1
            """,
            (pname,),
        ).fetchone()
        next_version = int(row["version"]) + 1 if row else 1
        snapshot_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO profile_snapshots(
                id, profile_name, version, static_json, dynamic_json,
                source_window_start, source_window_end, created_at, created_by
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                pname,
                next_version,
                json.dumps(static_items, ensure_ascii=False, separators=(",", ":")),
                json.dumps(dynamic_items, ensure_ascii=False, separators=(",", ":")),
                source_window_start,
                source_window_end,
                now,
                created_by[:40],
            ),
        )
        conn.commit()
    return {
        "id": snapshot_id,
        "profile_name": pname,
        "version": next_version,
        "created_at": now,
        "created_by": created_by[:40],
    }


def get_latest_profile_snapshot(
    db_path: Path,
    *,
    profile_name: str = "global",
) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, profile_name, version, static_json, dynamic_json,
                   source_window_start, source_window_end, created_at, created_by
            FROM profile_snapshots
            WHERE profile_name=?
            ORDER BY version DESC
            LIMIT 1
            """,
            (profile_name[:64],),
        ).fetchone()
    if not row:
        return None
    out = {k: row[k] for k in row.keys()}
    try:
        out["static_items"] = json.loads(out.pop("static_json"))
    except Exception:
        out["static_items"] = []
    try:
        out["dynamic_items"] = json.loads(out.pop("dynamic_json"))
    except Exception:
        out["dynamic_items"] = []
    return out


def get_memory_node_counts(db_path: Path) -> dict[str, int]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM memory_nodes
            GROUP BY status
            """
        ).fetchall()
    totals = {str(row["status"]): int(row["count"]) for row in rows}
    return {
        "active": totals.get("active", 0),
        "superseded": totals.get("superseded", 0),
        "forgotten": totals.get("forgotten", 0),
    }

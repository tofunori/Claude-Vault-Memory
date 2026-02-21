from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _turn_texts() -> list[str]:
    # Repeated durable fact with light paraphrases to drive conservative promotion.
    return [
        "On utilise FastAPI + SQLite WAL sur NAS pour la mémoire.",
        "La stack mémoire retenue: FastAPI avec queue SQLite WAL sur le NAS.",
        "Decision: garder FastAPI/SQLite WAL côté NAS pour memory-api.",
        "Contrainte: aucun traitement mémoire local sur Mac, seulement appels NAS.",
        "On confirme la stack mémoire NAS: FastAPI + SQLite WAL.",
        "Même choix technique: API mémoire FastAPI et queue SQLite WAL sur NAS.",
        "Préférence stable: architecture mémoire centralisée sur NAS.",
        "Toujours la même décision: FastAPI + SQLite WAL pour la mémoire.",
        "C'est validé: on garde FastAPI + SQLite WAL sur NAS.",
        "Décision finale: FastAPI + SQLite WAL sur NAS pour la couche mémoire unique.",
    ]


def _build_session_stop_conversation(turns: list[str]) -> str:
    return "\n\n".join([f"USER: {text}" for text in turns])


def run_synthetic_cycle(
    client: Any,
    *,
    run_id: str,
    cycle_index: int,
    expected_experimental_min: int = 1,
    retrieve_wait_timeout_s: int = 20,
    queue_settle_timeout_s: int = 45,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    turns = _turn_texts()
    session_id = f"burnin-synth-{run_id}-{cycle_index}-{int(time.time())}"
    foreign_session_id = f"burnin-synth-foreign-{run_id}-{cycle_index}-{int(time.time())}"

    turn_ack_failures = 0
    last_turn_http_status: int | None = None
    for idx, text in enumerate(turns, start=1):
        payload = {
            "event_type": "turn",
            "agent": "codex",
            "session_id": session_id,
            "payload": {
                "turn_index": idx,
                "role": "user",
                "text": text[:4000],
                "cwd": "/volume1/Services/memory",
                "ts": utc_now_iso(),
            },
        }
        resp = client.post("/events", payload)
        last_turn_http_status = resp.get("http_status")
        if not resp.get("ok"):
            turn_ack_failures += 1

    experimental_used = 0
    retrieve_ok = False
    deadline = time.time() + retrieve_wait_timeout_s
    while time.time() < deadline:
        resp = client.post(
            "/retrieve",
            {
                "query": "Quelle stack mémoire NAS a été décidée?",
                "session_id": session_id,
                "agent": "codex",
                "top_k": 5,
            },
        )
        data = resp.get("data") or {}
        if resp.get("ok") and data.get("status") == "ok":
            retrieve_ok = True
            diagnostics = data.get("diagnostics") or {}
            experimental_used = int(diagnostics.get("experimental_used") or 0)
            if experimental_used >= expected_experimental_min:
                break
        time.sleep(2)

    isolate_resp = client.post(
        "/retrieve",
        {
            "query": "Quelle stack mémoire NAS a été décidée?",
            "session_id": foreign_session_id,
            "agent": "codex",
            "top_k": 5,
        },
    )
    isolate_data = isolate_resp.get("data") or {}
    isolate_used = int(((isolate_data.get("diagnostics") or {}).get("experimental_used") or 0))
    isolation_ok = bool(isolate_resp.get("ok")) and isolate_used == 0

    stop_resp = client.post(
        "/events",
        {
            "event_type": "session_stop",
            "agent": "codex",
            "session_id": session_id,
            "payload": {
                "turn_count": len(turns),
                "cwd": "/volume1/Services/memory",
                "conversation_text": _build_session_stop_conversation(turns),
            },
        },
    )

    queue_stable = False
    queue_deadline = time.time() + queue_settle_timeout_s
    queue_wait_s = 0.0
    while time.time() < queue_deadline:
        health = client.get("/health")
        if health.get("ok"):
            counts = ((health.get("data") or {}).get("queue_db") or {}).get("counts") or {}
            queued = int(counts.get("queued", 0))
            processing = int(counts.get("processing", 0))
            if queued == 0 and processing == 0:
                queue_stable = True
                break
        time.sleep(2)
        queue_wait_s += 2

    experimental_ok = experimental_used >= expected_experimental_min
    success = (
        turn_ack_failures == 0
        and retrieve_ok
        and experimental_ok
        and isolation_ok
        and bool(stop_resp.get("ok"))
        and queue_stable
    )

    return {
        "ts": utc_now_iso(),
        "started_at": started_at,
        "cycle_index": cycle_index,
        "session_id": session_id,
        "foreign_session_id": foreign_session_id,
        "turn_ack_failures": turn_ack_failures,
        "last_turn_http_status": last_turn_http_status,
        "experimental_used": experimental_used,
        "experimental_ok": experimental_ok,
        "isolation_ok": isolation_ok,
        "isolation_experimental_used": isolate_used,
        "session_stop_ok": bool(stop_resp.get("ok")),
        "queue_stable": queue_stable,
        "queue_wait_s": queue_wait_s,
        "success": success,
    }


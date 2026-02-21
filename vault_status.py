#!/usr/bin/env python3
"""
vault_status.py — Health check and metrics dashboard for Claude Vault Memory.

Usage:
  python3 vault_status.py          → CLI dashboard
  python3 vault_status.py --json   → JSON output (for monitoring)
"""

import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR as _VND, QDRANT_PATH as _QP, ENV_FILE as _EF,
        LOG_FILE as _LF, QUEUE_DIR as _QD, GRAPH_CACHE_PATH as _GCP,
    )
    VAULT_NOTES_DIR = Path(_VND)
    QDRANT_PATH = Path(_QP)
    ENV_FILE = Path(_EF)
    LOG_FILE = Path(_LF)
    QUEUE_DIR = Path(_QD)
    GRAPH_CACHE_PATH = Path(_GCP)
except ImportError:
    print("ERROR: config.py not found.")
    sys.exit(1)

TODAY = date.today().isoformat()


def count_notes() -> dict:
    """Count notes and analyze vault health."""
    if not VAULT_NOTES_DIR.exists():
        return {"total": 0, "error": "vault directory not found"}

    total = 0
    by_type = {}
    by_confidence = {}
    orphans = 0  # Notes with no links
    oldest = None
    newest = None

    for p in VAULT_NOTES_DIR.glob("*.md"):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        total += 1
        try:
            text = p.read_text(encoding="utf-8")[:600]
            type_m = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
            conf_m = re.search(r'^confidence:\s*(.+)$', text, re.MULTILINE)
            created_m = re.search(r'^created:\s*(.+)$', text, re.MULTILINE)
            has_links = bool(re.search(r'\[\[', text))

            note_type = type_m.group(1).strip() if type_m else "unknown"
            confidence = conf_m.group(1).strip() if conf_m else "unknown"
            created = created_m.group(1).strip() if created_m else None

            by_type[note_type] = by_type.get(note_type, 0) + 1
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1

            if not has_links:
                orphans += 1

            if created:
                if oldest is None or created < oldest:
                    oldest = created
                if newest is None or created > newest:
                    newest = created
        except Exception:
            pass

    return {
        "total": total,
        "by_type": by_type,
        "by_confidence": by_confidence,
        "orphans": orphans,
        "oldest": oldest,
        "newest": newest,
    }


def check_services() -> dict:
    """Check health of external services."""
    services = {}

    # Qdrant
    services["qdrant"] = {
        "status": "ok" if QDRANT_PATH.exists() else "missing",
        "path": str(QDRANT_PATH),
    }

    # API keys
    env = {}
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass

    voyage_key = env.get("VOYAGE_API_KEY", "")
    fireworks_key = env.get("FIREWORKS_API_KEY", "")
    services["voyage_api"] = "configured" if voyage_key and not voyage_key.startswith("<") else "missing"
    services["fireworks_api"] = "configured" if fireworks_key and not fireworks_key.startswith("<") else "missing"

    # Graph cache
    if GRAPH_CACHE_PATH.exists():
        try:
            cache = json.loads(GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
            edge_count = sum(len(v) for v in cache.get("outbound", {}).values())
            services["graph_cache"] = {
                "status": "ok",
                "notes": cache.get("note_count", "?"),
                "edges": edge_count,
                "built_at": cache.get("built_at", "?"),
                "last_incremental": cache.get("last_incremental", "never"),
            }
        except Exception:
            services["graph_cache"] = {"status": "corrupted"}
    else:
        services["graph_cache"] = {"status": "missing"}

    return services


def analyze_log() -> dict:
    """Analyze recent log entries for health metrics."""
    if not LOG_FILE.exists():
        return {"error": "log file not found"}

    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {"error": "cannot read log file"}

    # Count today's events
    today_prefix = f"[{TODAY}]"
    today_lines = [l for l in lines if l.startswith(today_prefix)]

    retrievals = sum(1 for l in today_lines if "RETRIEVE" in l and "error" not in l.lower())
    retrieval_errors = sum(1 for l in today_lines if "RETRIEVE error" in l)
    sessions_processed = sum(1 for l in today_lines if "PROCESSING session" in l)
    facts_extracted = 0
    for l in today_lines:
        m = re.search(r'Facts extracted: (\d+)', l)
        if m:
            facts_extracted += int(m.group(1))
    dedup_count = sum(1 for l in today_lines if "DEDUP:" in l)
    graph_updates = sum(1 for l in today_lines if "GRAPH incremental" in l)
    embed_errors = sum(1 for l in today_lines if "error" in l.lower() and "EMBED" in l)

    # Queue status
    queue_pending = 0
    queue_processed = 0
    try:
        queue_pending = len(list(QUEUE_DIR.glob("*.json")))
        processed_dir = QUEUE_DIR / "processed"
        if processed_dir.exists():
            queue_processed = len(list(processed_dir.glob("*.json")))
    except Exception:
        pass

    return {
        "today": {
            "retrievals": retrievals,
            "retrieval_errors": retrieval_errors,
            "sessions_processed": sessions_processed,
            "facts_extracted": facts_extracted,
            "dedup_count": dedup_count,
            "graph_updates": graph_updates,
            "embed_errors": embed_errors,
        },
        "queue": {
            "pending": queue_pending,
            "processed_total": queue_processed,
        },
        "log_lines_total": len(lines),
    }


def main():
    json_mode = "--json" in sys.argv

    notes = count_notes()
    services = check_services()
    log_stats = analyze_log()

    report = {
        "timestamp": datetime.now().isoformat(),
        "notes": notes,
        "services": services,
        "activity": log_stats,
    }

    if json_mode:
        print(json.dumps(report, indent=2))
        return

    # Pretty CLI output
    print("=" * 60)
    print("  CLAUDE VAULT MEMORY — STATUS DASHBOARD")
    print("=" * 60)

    print(f"\n--- Notes ({notes['total']} total) ---")
    if notes.get("by_type"):
        for t, c in sorted(notes["by_type"].items(), key=lambda x: -x[1]):
            print(f"  {t:20s} {c:4d}")
    if notes.get("by_confidence"):
        print(f"\n  Confidence:")
        for c, n in sorted(notes["by_confidence"].items(), key=lambda x: -x[1]):
            print(f"    {c:20s} {n:4d}")
    print(f"  Orphans (no links):  {notes.get('orphans', '?')}")
    print(f"  Date range:          {notes.get('oldest', '?')} → {notes.get('newest', '?')}")

    print(f"\n--- Services ---")
    print(f"  Qdrant:              {services['qdrant']['status']}")
    print(f"  Voyage API:          {services['voyage_api']}")
    print(f"  Fireworks API:       {services['fireworks_api']}")
    gc = services.get("graph_cache", {})
    if isinstance(gc, dict):
        print(f"  Graph cache:         {gc.get('status', '?')} ({gc.get('notes', '?')} notes, {gc.get('edges', '?')} edges)")
        print(f"    Built:             {gc.get('built_at', '?')}")
        print(f"    Last incremental:  {gc.get('last_incremental', 'never')}")

    activity = log_stats.get("today", {})
    print(f"\n--- Today's Activity ---")
    print(f"  Retrievals:          {activity.get('retrievals', 0)} ({activity.get('retrieval_errors', 0)} errors)")
    print(f"  Sessions processed:  {activity.get('sessions_processed', 0)}")
    print(f"  Facts extracted:     {activity.get('facts_extracted', 0)}")
    print(f"  Dedup merges:        {activity.get('dedup_count', 0)}")
    print(f"  Graph updates:       {activity.get('graph_updates', 0)}")

    queue = log_stats.get("queue", {})
    print(f"\n--- Queue ---")
    print(f"  Pending:             {queue.get('pending', 0)}")
    print(f"  Processed (total):   {queue.get('processed_total', 0)}")

    # Health warnings
    warnings = []
    if services["voyage_api"] == "missing":
        warnings.append("Voyage API key not configured — retrieval disabled")
    if services["fireworks_api"] == "missing":
        warnings.append("Fireworks API key not configured — extraction disabled")
    if services["qdrant"]["status"] == "missing":
        warnings.append("Qdrant index missing — run: python3 vault_embed.py")
    if isinstance(gc, dict) and gc.get("status") == "missing":
        warnings.append("Graph cache missing — run: python3 vault_embed.py")
    if notes.get("orphans", 0) > notes["total"] * 0.3 and notes["total"] > 10:
        warnings.append(f"{notes['orphans']} orphan notes ({int(notes['orphans']/notes['total']*100)}%) — consider adding links")
    if activity.get("retrieval_errors", 0) > 5:
        warnings.append(f"High retrieval error rate today ({activity['retrieval_errors']} errors)")

    if warnings:
        print(f"\n--- Warnings ---")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print(f"\n  All systems healthy.")

    print()


if __name__ == "__main__":
    main()

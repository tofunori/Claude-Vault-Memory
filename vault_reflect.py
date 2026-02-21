#!/usr/bin/env python3
"""
vault_reflect.py — Periodic reflector for vault maintenance (v2).

Inspired by Observational Memory's reflector. Run periodically (weekly cron).

What it does:
1. Finds clusters of semantically similar notes and suggests merges
2. Flags stale notes (never retrieved, old) for review
3. Flags orphan notes (no incoming or outgoing links)
4. Auto-archives expired notes (forget_after date passed)
5. Generates a reflection report

Usage:
  python3 vault_reflect.py              → generate report (dry run)
  python3 vault_reflect.py --apply      → apply automatic actions (mark stale, archive expired)
  python3 vault_reflect.py --json       → JSON output

Recommended cron: weekly on Sundays
  0 4 * * 0 python3 /path/to/vault_reflect.py --apply >> /path/to/auto_remember.log 2>&1
"""

import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR as _VND, QDRANT_PATH as _QP, ENV_FILE as _EF,
        LOG_FILE as _LF, GRAPH_CACHE_PATH as _GCP, VOYAGE_EMBED_MODEL,
    )
    VAULT_NOTES_DIR = Path(_VND)
    QDRANT_PATH = Path(_QP)
    ENV_FILE = Path(_EF)
    LOG_FILE = Path(_LF)
    GRAPH_CACHE_PATH = Path(_GCP)
except ImportError:
    print("ERROR: config.py not found.")
    sys.exit(1)

try:
    from config import REFLECT_MIN_NOTES
except ImportError:
    REFLECT_MIN_NOTES = 30
try:
    from config import REFLECT_CLUSTER_THRESHOLD
except ImportError:
    REFLECT_CLUSTER_THRESHOLD = 0.82
try:
    from config import REFLECT_STALE_DAYS
except ImportError:
    REFLECT_STALE_DAYS = 180

try:
    from config import FORGET_ARCHIVE_DIR as _FAD
    FORGET_ARCHIVE_DIR = Path(_FAD)
except ImportError:
    FORGET_ARCHIVE_DIR = None  # Will be set from VAULT_NOTES_DIR

try:
    from config import FORGET_DEFAULT_TTL_DAYS
except ImportError:
    FORGET_DEFAULT_TTL_DAYS = {}  # e.g. {"context": 90, "result": 60}

COLLECTION = "vault_notes"
TODAY = date.today()


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{TODAY.isoformat()}] {msg}\n")
    except Exception:
        pass


def load_env_file() -> dict:
    env = {}
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def parse_frontmatter(path: Path) -> dict:
    """Parse note frontmatter + basic stats."""
    try:
        text = path.read_text(encoding="utf-8")[:800]
    except Exception:
        return {}

    fm = {"note_id": path.stem, "path": str(path)}
    for field in ("description", "type", "confidence", "created", "forget_after",
                  "stale", "relation", "parent_note", "superseded_by"):
        m = re.search(rf'^{field}:\s*(.+)$', text, re.MULTILINE)
        if m:
            fm[field] = m.group(1).strip()

    # Count links
    links = re.findall(r'\[\[([^\]]+)\]\]', text)
    fm["outbound_links"] = len(links)
    fm["char_count"] = len(text)
    return fm


def find_similar_clusters(notes: list[dict]) -> list[list[str]]:
    """Find clusters of semantically similar notes using Qdrant."""
    try:
        import voyageai
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchAny
    except ImportError:
        return []

    env = load_env_file()
    api_key = env.get("VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key.startswith("<") or not QDRANT_PATH.exists():
        return []

    try:
        qd = QdrantClient(path=str(QDRANT_PATH))
        existing = {c.name for c in qd.get_collections().collections}
        if COLLECTION not in existing:
            return []
    except Exception:
        return []

    clusters = []
    clustered = set()
    note_ids = [n["note_id"] for n in notes]

    for note in notes:
        nid = note["note_id"]
        if nid in clustered:
            continue

        try:
            import uuid
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, nid))

            # Get this note's vector
            points = qd.retrieve(collection_name=COLLECTION, ids=[point_id], with_vectors=True)
            if not points:
                continue

            vector = points[0].vector

            # Find similar notes
            response = qd.query_points(
                collection_name=COLLECTION,
                query=vector,
                limit=5,
                score_threshold=REFLECT_CLUSTER_THRESHOLD,
            )

            similar = [
                r.payload["note_id"]
                for r in response.points
                if r.payload["note_id"] != nid and r.payload["note_id"] not in clustered
            ]

            if similar:
                cluster = [nid] + similar
                clusters.append(cluster)
                clustered.update(cluster)

        except Exception:
            continue

    return clusters


def find_stale_notes(notes: list[dict]) -> list[dict]:
    """Find notes that are old and have never been retrieved."""
    cutoff = (TODAY - timedelta(days=REFLECT_STALE_DAYS)).isoformat()
    stale = []

    try:
        from qdrant_client import QdrantClient
        import uuid
        qd = QdrantClient(path=str(QDRANT_PATH))

        for note in notes:
            nid = note["note_id"]
            created = note.get("created", TODAY.isoformat())

            if created > cutoff:
                continue  # Too recent

            # Check last_retrieved in Qdrant
            try:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, nid))
                points = qd.retrieve(collection_name=COLLECTION, ids=[point_id])
                if points:
                    last_ret = points[0].payload.get("last_retrieved", created)
                    if last_ret <= cutoff:
                        note["last_retrieved"] = last_ret
                        stale.append(note)
            except Exception:
                continue
    except Exception:
        # Fallback: just check created date
        for note in notes:
            created = note.get("created", TODAY.isoformat())
            if created <= cutoff:
                stale.append(note)

    return stale


def find_expired_notes(notes: list[dict]) -> list[dict]:
    """Find notes whose forget_after date has passed, or whose type-based TTL has expired."""
    expired = []
    today_str = TODAY.isoformat()

    for note in notes:
        forget_after = note.get("forget_after")

        # Check explicit forget_after date
        if forget_after and forget_after <= today_str:
            note["expiry_reason"] = f"forget_after: {forget_after}"
            expired.append(note)
            continue

        # Check type-based default TTL (if configured and no explicit forget_after)
        if not forget_after and FORGET_DEFAULT_TTL_DAYS:
            ntype = note.get("type", "")
            ttl_days = FORGET_DEFAULT_TTL_DAYS.get(ntype)
            if ttl_days and note.get("created"):
                try:
                    created = date.fromisoformat(note["created"][:10])
                    expiry = created + timedelta(days=ttl_days)
                    if TODAY >= expiry:
                        note["expiry_reason"] = f"type '{ntype}' TTL: {ttl_days} days"
                        expired.append(note)
                except (ValueError, TypeError):
                    pass

    return expired


def archive_expired_notes(expired: list[dict]):
    """Move expired notes to _archived/ directory and remove from Qdrant."""
    archive_dir = FORGET_ARCHIVE_DIR or (VAULT_NOTES_DIR / "_archived")
    archive_dir.mkdir(parents=True, exist_ok=True)

    archived = 0
    for note in expired:
        note_path = Path(note["path"])
        if not note_path.exists():
            continue
        try:
            dest = archive_dir / note_path.name
            note_path.rename(dest)
            archived += 1
            log(f"FORGET archived: {note['note_id']} ({note.get('expiry_reason', '?')})")
        except Exception as e:
            log(f"FORGET archive error for {note['note_id']}: {e}")

    # Remove from Qdrant
    if archived > 0:
        try:
            import uuid as _uuid
            from qdrant_client import QdrantClient
            qd = QdrantClient(path=str(QDRANT_PATH))
            point_ids = [
                str(_uuid.uuid5(_uuid.NAMESPACE_DNS, n["note_id"]))
                for n in expired
            ]
            qd.delete(collection_name=COLLECTION, points_selector=point_ids)
            log(f"FORGET removed {len(point_ids)} points from Qdrant")
        except Exception as e:
            log(f"FORGET Qdrant cleanup error: {e}")

    return archived


def find_orphan_notes(notes: list[dict]) -> list[str]:
    """Find notes with no incoming or outgoing links."""
    if not GRAPH_CACHE_PATH.exists():
        return []

    try:
        cache = json.loads(GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
        outbound = cache.get("outbound", {})
        backlinks = cache.get("backlinks", {})
    except Exception:
        return []

    orphans = []
    for note in notes:
        nid = note["note_id"]
        has_outbound = bool(outbound.get(nid, []))
        has_backlinks = bool(backlinks.get(nid, []))
        if not has_outbound and not has_backlinks:
            orphans.append(nid)

    return orphans


def main():
    apply_mode = "--apply" in sys.argv
    json_mode = "--json" in sys.argv

    # Scan all notes
    notes = []
    for p in sorted(VAULT_NOTES_DIR.glob("*.md")):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        fm = parse_frontmatter(p)
        if fm:
            notes.append(fm)

    total = len(notes)

    if total < REFLECT_MIN_NOTES:
        msg = f"REFLECT: vault too small ({total} notes, min {REFLECT_MIN_NOTES}). Skipping."
        log(msg)
        if not json_mode:
            print(msg)
        return

    log(f"=== REFLECT: analyzing {total} notes ===")

    # 1. Find expired notes (forget_after passed or type TTL exceeded)
    expired = find_expired_notes(notes)

    # 2. Find similar clusters
    clusters = find_similar_clusters(notes)

    # 3. Find stale notes
    stale = find_stale_notes(notes)

    # 4. Find orphan notes
    orphans = find_orphan_notes(notes)

    report = {
        "timestamp": TODAY.isoformat(),
        "total_notes": total,
        "expired_notes": [
            {"note_id": e["note_id"], "reason": e.get("expiry_reason", "?")}
            for e in expired
        ],
        "clusters": [
            {"notes": c, "action": "review_merge"}
            for c in clusters
        ],
        "stale_notes": [
            {"note_id": s["note_id"], "created": s.get("created", "?"), "last_retrieved": s.get("last_retrieved", "never")}
            for s in stale
        ],
        "orphan_notes": orphans,
        "summary": {
            "expired_count": len(expired),
            "clusters_found": len(clusters),
            "stale_count": len(stale),
            "orphan_count": len(orphans),
        }
    }

    if json_mode:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 60)
        print("  VAULT REFLECTOR — Analysis Report")
        print("=" * 60)
        print(f"\n  Total notes: {total}")

        if expired:
            print(f"\n--- Expired Notes ({len(expired)}) ---")
            print("  These notes have passed their forget_after date or type TTL:")
            for e in expired[:20]:
                print(f"    - {e['note_id']} ({e.get('expiry_reason', '?')})")
            if len(expired) > 20:
                print(f"    ... and {len(expired) - 20} more")
            if apply_mode:
                print(f"  → Will be archived to _archived/")
        else:
            print(f"\n  No expired notes found")

        if clusters:
            print(f"\n--- Similar Note Clusters ({len(clusters)}) ---")
            print("  These notes are semantically very close and may be candidates for merging:")
            for i, c in enumerate(clusters, 1):
                print(f"\n  Cluster {i}:")
                for nid in c:
                    desc = next((n.get("description", "") for n in notes if n["note_id"] == nid), "")
                    print(f"    - {nid}: {desc[:80]}")
        else:
            print(f"\n  No similar clusters found (threshold: {REFLECT_CLUSTER_THRESHOLD})")

        if stale:
            print(f"\n--- Stale Notes ({len(stale)}) ---")
            print(f"  Not retrieved in {REFLECT_STALE_DAYS}+ days:")
            for s in stale[:20]:
                print(f"    - {s['note_id']} (created: {s.get('created', '?')}, last: {s.get('last_retrieved', 'never')})")
            if len(stale) > 20:
                print(f"    ... and {len(stale) - 20} more")
        else:
            print(f"\n  No stale notes found (threshold: {REFLECT_STALE_DAYS} days)")

        if orphans:
            print(f"\n--- Orphan Notes ({len(orphans)}) ---")
            print("  No incoming or outgoing links:")
            for o in orphans[:20]:
                print(f"    - {o}")
            if len(orphans) > 20:
                print(f"    ... and {len(orphans) - 20} more")
        else:
            print(f"\n  No orphan notes found")

        # Recommendations
        print(f"\n--- Recommendations ---")
        rec = 0
        if expired:
            rec += 1
            if apply_mode:
                print(f"  {rec}. Archiving {len(expired)} expired note(s) automatically")
            else:
                print(f"  {rec}. {len(expired)} expired note(s) ready to archive (run --apply)")
        if clusters:
            rec += 1
            print(f"  {rec}. Review {len(clusters)} cluster(s) for potential merging")
        if stale:
            rec += 1
            print(f"  {rec}. Review {len(stale)} stale note(s) — consider archiving or updating")
        if orphans:
            rec += 1
            print(f"  {rec}. Add [[links]] to {len(orphans)} orphan note(s) to integrate them into the graph")
        if not clusters and not stale and not orphans and not expired:
            print("  Vault is healthy. No action needed.")
        print()

    # Apply actions (if --apply)
    if apply_mode:
        # 1. Archive expired notes (forget_after passed)
        if expired:
            archived_count = archive_expired_notes(expired)
            log(f"REFLECT: archived {archived_count} expired notes")

        # 2. Mark stale notes
        if stale:
            marked = 0
            for s in stale:
                note_path = VAULT_NOTES_DIR / f"{s['note_id']}.md"
                if note_path.exists():
                    try:
                        content = note_path.read_text(encoding="utf-8")
                        if "stale: true" not in content:
                            content = content.replace(
                                "---\n\n",
                                f"stale: true\nstale_since: {TODAY.isoformat()}\n---\n\n",
                                1,
                            )
                            note_path.write_text(content, encoding="utf-8")
                            marked += 1
                    except Exception:
                        pass
            if marked:
                log(f"REFLECT: marked {marked} notes as stale")

    log(f"REFLECT: {len(expired)} expired, {len(clusters)} clusters, {len(stale)} stale, {len(orphans)} orphans")


if __name__ == "__main__":
    main()

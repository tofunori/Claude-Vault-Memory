#!/usr/bin/env python3
"""
vault_embed.py — Build/upsert local Qdrant index from vault notes (v6).

Usage:
  python3 vault_embed.py              → full rebuild (all notes)
  python3 vault_embed.py --note ID    → incremental upsert (single note)
  python3 vault_embed.py --notes A B  → incremental upsert (list of notes)

v6: stores confidence + last_retrieved in Qdrant payload for decay/weighting.
"""

import os
import re
import sys
import uuid
from datetime import date
from pathlib import Path

# Load config from same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR as _VAULT_NOTES_DIR,
        QDRANT_PATH as _QDRANT_PATH,
        ENV_FILE as _ENV_FILE,
        LOG_FILE as _LOG_FILE,
        VOYAGE_EMBED_MODEL,
        EMBED_DIM,
        EMBED_BATCH_SIZE,
        GRAPH_CACHE_PATH as _GRAPH_CACHE_PATH,
    )
    VAULT_NOTES_DIR = Path(_VAULT_NOTES_DIR)
    QDRANT_PATH = Path(_QDRANT_PATH)
    ENV_FILE = Path(_ENV_FILE)
    LOG_FILE = Path(_LOG_FILE)
    GRAPH_CACHE_PATH = Path(_GRAPH_CACHE_PATH)
except ImportError:
    VAULT_NOTES_DIR = Path.home() / "notes"
    QDRANT_PATH = Path.home() / ".claude/hooks/vault_qdrant"
    ENV_FILE = Path.home() / ".claude/hooks/.env"
    LOG_FILE = Path.home() / ".claude/hooks/auto_remember.log"
    GRAPH_CACHE_PATH = Path.home() / ".claude/hooks/vault_graph_cache.json"
    VOYAGE_EMBED_MODEL = "voyage-4-large"
    EMBED_DIM = 1024
    EMBED_BATCH_SIZE = 128

COLLECTION = "vault_notes"
TODAY = date.today().isoformat()


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{TODAY}] {msg}\n")
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


def get_clients():
    """Initialize Voyage AI and Qdrant. Creates collection if missing."""
    try:
        import voyageai
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError as e:
        log(f"EMBED import error: {e} — install: pip install voyageai qdrant-client")
        sys.exit(1)

    env = load_env_file()
    api_key = env.get("VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        log("EMBED SKIP: VOYAGE_API_KEY missing or placeholder in .env")
        sys.exit(0)

    vo = voyageai.Client(api_key=api_key)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qd = QdrantClient(path=str(QDRANT_PATH))

    existing = {c.name for c in qd.get_collections().collections}
    if COLLECTION not in existing:
        qd.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        log(f"EMBED collection created: {COLLECTION} (dim={EMBED_DIM})")

    return vo, qd


def parse_note(path: Path) -> dict | None:
    """Extract text and metadata from a markdown note."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    desc_m = re.search(r'^description:\s*(.+)$', text, re.MULTILINE)
    type_m = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
    title_m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    created_m = re.search(r'^created:\s*(.+)$', text, re.MULTILINE)
    confidence_m = re.search(r'^confidence:\s*(.+)$', text, re.MULTILINE)

    description = desc_m.group(1).strip() if desc_m else (title_m.group(1).strip() if title_m else path.stem)
    note_type = type_m.group(1).strip() if type_m else "concept"
    created = created_m.group(1).strip() if created_m else TODAY
    confidence = confidence_m.group(1).strip() if confidence_m else "experimental"

    body = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL).strip()
    embed_text = f"{description}\n\n{body}"[:4000]  # voyage-4-large handles long context

    return {
        "note_id": path.stem,
        "text": embed_text,
        "description": description,
        "type": note_type,
        "created": created,
        "confidence": confidence,
    }


def get_notes_to_embed(note_ids: list[str] | None = None) -> list[dict]:
    notes = []
    if note_ids:
        for nid in note_ids:
            p = VAULT_NOTES_DIR / f"{nid}.md"
            if p.exists():
                n = parse_note(p)
                if n:
                    notes.append(n)
            else:
                log(f"EMBED WARN: note not found: {nid}")
    else:
        for p in sorted(VAULT_NOTES_DIR.glob("*.md")):
            if p.name.startswith(".") or p.name.startswith("_"):
                continue
            n = parse_note(p)
            if n:
                notes.append(n)
    return notes


def build_graph_index(notes: list[dict]) -> tuple[dict, dict]:
    """Build outbound link index and backlink index from already-parsed notes."""
    known_ids = {n["note_id"] for n in notes}
    outbound: dict[str, list[str]] = {}
    for n in notes:
        links = re.findall(r'\[\[([^\]]+)\]\]', n["text"])
        outbound[n["note_id"]] = list(dict.fromkeys(
            l.strip() for l in links
            if len(l.strip()) < 60 and ' ' not in l.strip() and l.strip() in known_ids
        ))
    backlinks: dict[str, list[str]] = {}
    for src, targets in outbound.items():
        for t in targets:
            backlinks.setdefault(t, []).append(src)
    return outbound, backlinks


def upsert_notes(note_ids: list[str] | None = None):
    try:
        from qdrant_client.models import PointStruct
    except ImportError:
        log("EMBED import PointStruct failed")
        sys.exit(1)

    vo, qd = get_clients()
    notes = get_notes_to_embed(note_ids)

    if not notes:
        log("EMBED: no notes to upsert")
        return

    total = 0
    for i in range(0, len(notes), EMBED_BATCH_SIZE):
        batch = notes[i:i + EMBED_BATCH_SIZE]
        texts = [n["text"] for n in batch]

        try:
            result = vo.embed(
                texts,
                model=VOYAGE_EMBED_MODEL,
                input_type="document",
                truncation=True,
            )
            embeddings = result.embeddings
        except Exception as e:
            log(f"EMBED Voyage AI API error (batch {i}): {e}")
            continue

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, n["note_id"])),
                vector=emb,
                payload={
                    "note_id": n["note_id"],
                    "description": n["description"],
                    "type": n["type"],
                    "created": n["created"],
                    "confidence": n["confidence"],
                    "last_retrieved": n["created"],  # Initialize to created date
                    "updated_at": TODAY,
                }
            )
            for n, emb in zip(batch, embeddings)
        ]

        try:
            qd.upsert(collection_name=COLLECTION, points=points)
            total += len(points)
        except Exception as e:
            log(f"EMBED Qdrant upsert error (batch {i}): {e}")

    log(f"EMBED_INDEX upserted: {total} notes")
    if note_ids is None:
        print(f"EMBED_INDEX upserted: {total} notes → {QDRANT_PATH}")
        # Rebuild graph cache from parsed notes (no extra I/O needed)
        try:
            import json
            outbound, backlinks = build_graph_index(notes)
            cache = {
                "built_at": TODAY,
                "note_count": len(notes),
                "outbound": outbound,
                "backlinks": backlinks,
            }
            GRAPH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            edge_count = sum(len(v) for v in outbound.values())
            log(f"EMBED graph cache: {len(outbound)} notes, {edge_count} edges")
            print(f"EMBED graph cache: {len(outbound)} notes, {edge_count} edges → {GRAPH_CACHE_PATH}")
        except Exception as e:
            log(f"EMBED graph cache error: {e}")


def main():
    args = sys.argv[1:]

    if "--note" in args:
        idx = args.index("--note")
        note_id = args[idx + 1] if idx + 1 < len(args) else None
        if not note_id:
            print("Usage: vault_embed.py --note NOTE_ID")
            sys.exit(1)
        upsert_notes([note_id])

    elif "--notes" in args:
        idx = args.index("--notes")
        note_ids = args[idx + 1:]
        if not note_ids:
            print("Usage: vault_embed.py --notes NOTE_ID1 NOTE_ID2 ...")
            sys.exit(1)
        upsert_notes(note_ids)

    else:
        upsert_notes(None)


if __name__ == "__main__":
    main()

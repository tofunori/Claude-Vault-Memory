#!/usr/bin/env python3
"""
vault_embed.py — Build/upsert index Qdrant local depuis les notes du vault.

Usage:
  python3 vault_embed.py              → full rebuild (toutes les notes)
  python3 vault_embed.py --note ID    → upsert incrémental (une note)
  python3 vault_embed.py --notes A B  → upsert incrémental (liste de notes)
"""

import os
import re
import sys
import uuid
from datetime import date
from pathlib import Path

try:
    from config import (
        VAULT_NOTES_DIR as _VAULT_NOTES_DIR,
        QDRANT_PATH as _QDRANT_PATH,
        ENV_FILE as _ENV_FILE,
        LOG_FILE as _LOG_FILE,
        COHERE_EMBED_MODEL,
        EMBED_DIM,
        COHERE_BATCH_SIZE,
    )
    VAULT_NOTES_DIR = Path(_VAULT_NOTES_DIR)
    QDRANT_PATH = Path(_QDRANT_PATH)
    ENV_FILE = Path(_ENV_FILE)
    LOG_FILE = Path(_LOG_FILE)
except ImportError:
    VAULT_NOTES_DIR = Path.home() / "notes"
    QDRANT_PATH = Path.home() / ".claude/hooks/vault_qdrant"
    ENV_FILE = Path.home() / ".claude/hooks/.env"
    LOG_FILE = Path.home() / ".claude/hooks/auto_remember.log"
    COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
    EMBED_DIM = 1024
    COHERE_BATCH_SIZE = 96

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
    """Initialise Cohere et Qdrant. Crée la collection si absente."""
    try:
        import cohere
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError as e:
        log(f"EMBED import error: {e} — installer: pip install cohere qdrant-client")
        sys.exit(1)

    env = load_env_file()
    api_key = env.get("COHERE_API_KEY") or os.environ.get("COHERE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        log("EMBED SKIP: COHERE_API_KEY absent ou placeholder dans .env")
        sys.exit(0)

    co = cohere.ClientV2(api_key)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qd = QdrantClient(path=str(QDRANT_PATH))

    existing = {c.name for c in qd.get_collections().collections}
    if COLLECTION not in existing:
        qd.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        log(f"EMBED collection créée: {COLLECTION}")

    return co, qd


def parse_note(path: Path) -> dict | None:
    """Extrait text et metadata d'une note markdown."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    desc_m = re.search(r'^description:\s*(.+)$', text, re.MULTILINE)
    type_m = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
    title_m = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    created_m = re.search(r'^created:\s*(.+)$', text, re.MULTILINE)

    description = desc_m.group(1).strip() if desc_m else (title_m.group(1).strip() if title_m else path.stem)
    note_type = type_m.group(1).strip() if type_m else "concept"
    created = created_m.group(1).strip() if created_m else TODAY

    body = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL).strip()
    embed_text = f"{description}\n\n{body}"[:2000]

    return {
        "note_id": path.stem,
        "text": embed_text,
        "description": description,
        "type": note_type,
        "created": created,
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
                log(f"EMBED WARN: note non trouvée: {nid}")
    else:
        for p in sorted(VAULT_NOTES_DIR.glob("*.md")):
            if p.name.startswith(".") or p.name.startswith("_"):
                continue
            n = parse_note(p)
            if n:
                notes.append(n)
    return notes


def upsert_notes(note_ids: list[str] | None = None):
    try:
        from qdrant_client.models import PointStruct
    except ImportError:
        log("EMBED import PointStruct failed")
        sys.exit(1)

    co, qd = get_clients()
    notes = get_notes_to_embed(note_ids)

    if not notes:
        log("EMBED: aucune note à upsert")
        return

    total = 0
    for i in range(0, len(notes), COHERE_BATCH_SIZE):
        batch = notes[i:i + COHERE_BATCH_SIZE]
        texts = [n["text"] for n in batch]

        try:
            response = co.embed(
                model=COHERE_EMBED_MODEL,
                texts=texts,
                input_type="search_document",
                embedding_types=["float"],
            )
            embeddings = response.embeddings.float_
        except Exception as e:
            log(f"EMBED Cohere API error (batch {i}): {e}")
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

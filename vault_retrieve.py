#!/usr/bin/env python3
"""
vault_retrieve.py — UserPromptSubmit hook, retrieval actif.

Input stdin : JSON Claude Code {"prompt": "...", "session_id": "...", ...}
Output      : texte injecté dans le contexte Claude (notes pertinentes)
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

try:
    from config import (
        QDRANT_PATH as _QDRANT_PATH,
        ENV_FILE as _ENV_FILE,
        LOG_FILE as _LOG_FILE,
        COHERE_EMBED_MODEL,
        RETRIEVE_SCORE_THRESHOLD,
        RETRIEVE_TOP_K,
        MIN_QUERY_LENGTH,
    )
    QDRANT_PATH = Path(_QDRANT_PATH)
    ENV_FILE = Path(_ENV_FILE)
    LOG_FILE = Path(_LOG_FILE)
except ImportError:
    QDRANT_PATH = Path.home() / ".claude/hooks/vault_qdrant"
    ENV_FILE = Path.home() / ".claude/hooks/.env"
    LOG_FILE = Path.home() / ".claude/hooks/auto_remember.log"
    COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
    RETRIEVE_SCORE_THRESHOLD = 0.60
    RETRIEVE_TOP_K = 3
    MIN_QUERY_LENGTH = 20

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


def main():
    try:
        raw = sys.stdin.read().strip()
        data = json.loads(raw) if raw else {}
    except Exception:
        sys.exit(0)

    query = data.get("prompt", "").strip()

    if len(query) < MIN_QUERY_LENGTH:
        sys.exit(0)

    if not QDRANT_PATH.exists():
        sys.exit(0)

    env = load_env_file()
    api_key = env.get("COHERE_API_KEY") or os.environ.get("COHERE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        sys.exit(0)

    try:
        import cohere
        from qdrant_client import QdrantClient
    except ImportError:
        sys.exit(0)

    try:
        co = cohere.ClientV2(api_key)
        qd = QdrantClient(path=str(QDRANT_PATH))

        existing = {c.name for c in qd.get_collections().collections}
        if COLLECTION not in existing:
            sys.exit(0)

        resp = co.embed(
            model=COHERE_EMBED_MODEL,
            texts=[query[:512]],
            input_type="search_query",
            embedding_types=["float"],
        )
        query_emb = resp.embeddings.float_[0]

        response = qd.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=RETRIEVE_TOP_K,
            score_threshold=RETRIEVE_SCORE_THRESHOLD,
        )
        results = response.points

        if not results:
            sys.exit(0)

        lines = ["=== Notes vault pertinentes ==="]
        for r in results:
            p = r.payload
            score_pct = int(r.score * 100)
            lines.append(
                f"[[{p['note_id']}]] ({p.get('type', '?')}, {score_pct}%) — {p.get('description', '')}"
            )
        print("\n".join(lines))

        log(f"RETRIEVE query={len(query)}c → {len(results)} notes (seuil {RETRIEVE_SCORE_THRESHOLD})")

    except Exception as e:
        log(f"RETRIEVE error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()

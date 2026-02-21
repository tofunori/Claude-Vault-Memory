#!/usr/bin/env python3
"""
vault_retrieve.py — UserPromptSubmit hook, active retrieval.

Input stdin : JSON from Claude Code {"prompt": "...", "session_id": "...", ...}
Output      : text injected into Claude context (relevant notes)
"""

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

# Load config from same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import (
        VAULT_NOTES_DIR, QDRANT_PATH, ENV_FILE, LOG_FILE,
        RETRIEVE_SCORE_THRESHOLD as SCORE_THRESHOLD,
        RETRIEVE_TOP_K as TOP_K,
        MIN_QUERY_LENGTH,
        VOYAGE_EMBED_MODEL,
        GRAPH_CACHE_PATH as _GRAPH_CACHE_PATH,
        MAX_SECONDARY,
        MAX_BACKLINKS_PER_NOTE,
        BFS_DEPTH,
    )
    VAULT_NOTES_DIR = Path(VAULT_NOTES_DIR)
    QDRANT_PATH = Path(QDRANT_PATH)
    ENV_FILE = Path(ENV_FILE)
    LOG_FILE = Path(LOG_FILE)
    GRAPH_CACHE_PATH = Path(_GRAPH_CACHE_PATH)
except ImportError:
    print("ERROR: config.py not found. Copy config.example.py to config.py and edit paths.", file=sys.stderr)
    sys.exit(0)

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


def load_graph_cache() -> tuple[dict, dict]:
    """Load pre-computed graph indices. Returns ({}, {}) on failure (graceful degradation)."""
    try:
        import json
        data = json.loads(GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("outbound", {}), data.get("backlinks", {})
    except Exception:
        return {}, {}


def collect_bfs_candidates(
    primary_ids: list[str],
    outbound: dict,
    backlinks: dict,
) -> list[str]:
    """
    Collect candidate connected notes via 2-level BFS + backlinks.
    Round-robin across primaries for diversification.
    Returns deduplicated list of candidate IDs (primaries excluded).
    """
    seen = set(primary_ids)
    candidates: list[str] = []

    def add(nid: str) -> bool:
        if nid not in seen:
            seen.add(nid)
            candidates.append(nid)
            return True
        return False

    # Backlinks of primary notes (injected first — structural relevance)
    for pid in primary_ids:
        for nid in backlinks.get(pid, [])[:MAX_BACKLINKS_PER_NOTE]:
            add(nid)

    # BFS depth 1: outbound links, round-robin across primaries
    depth1_lists = [outbound.get(pid, []) for pid in primary_ids]
    depth1_frontier: list[str] = []
    for i in range(max((len(x) for x in depth1_lists), default=0)):
        for links in depth1_lists:
            if i < len(links) and add(links[i]):
                depth1_frontier.append(links[i])

    # BFS depth 2: outbound links of depth-1 nodes
    depth2_lists = [outbound.get(nid, []) for nid in depth1_frontier]
    for i in range(max((len(x) for x in depth2_lists), default=0)):
        for links in depth2_lists:
            if i < len(links):
                add(links[i])

    return candidates


def score_candidates_qdrant(
    candidate_ids: list[str],
    query_emb: list,
    qd,
) -> list[dict]:
    """Rank candidate notes by cosine similarity to query via Qdrant filter query."""
    if not candidate_ids:
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        f = Filter(must=[FieldCondition(key="note_id", match=MatchAny(any=candidate_ids))])
        response = qd.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            query_filter=f,
            limit=MAX_SECONDARY,
        )
        return [
            {
                "note_id": r.payload["note_id"],
                "description": r.payload.get("description", r.payload["note_id"]),
                "type": r.payload.get("type", "?"),
                "score": r.score,
            }
            for r in response.points
        ]
    except Exception:
        return []


def pad_unscored(
    scored_ids: set,
    candidate_ids: list[str],
    slots: int,
) -> list[dict]:
    """Fallback for candidates absent from Qdrant (new notes not yet indexed)."""
    result = []
    for nid in candidate_ids:
        if nid in scored_ids or len(result) >= slots:
            break
        note_path = VAULT_NOTES_DIR / f"{nid}.md"
        if not note_path.exists():
            continue
        try:
            text = note_path.read_text(encoding="utf-8", errors="replace")[:400]
            desc_m = re.search(r'^description:\s*(.+)$', text, re.MULTILINE)
            type_m = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
            result.append({
                "note_id": nid,
                "description": desc_m.group(1).strip() if desc_m else nid,
                "type": type_m.group(1).strip() if type_m else "?",
                "score": None,
            })
        except Exception:
            pass
    return result


def main():
    # Read stdin
    try:
        raw = sys.stdin.read().strip()
        data = json.loads(raw) if raw else {}
    except Exception:
        sys.exit(0)

    query = data.get("prompt", "").strip()

    # Guard: message too short
    if len(query) < MIN_QUERY_LENGTH:
        sys.exit(0)

    # Guard: Qdrant index not built yet
    if not QDRANT_PATH.exists():
        sys.exit(0)

    # Guard: VOYAGE_API_KEY missing
    env = load_env_file()
    api_key = env.get("VOYAGE_API_KEY") or os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        sys.exit(0)

    # Runtime imports (fail silently if packages missing)
    try:
        import voyageai
        from qdrant_client import QdrantClient
    except ImportError:
        sys.exit(0)

    try:
        vo = voyageai.Client(api_key=api_key)
        qd = QdrantClient(path=str(QDRANT_PATH))

        # Check collection exists
        existing = {c.name for c in qd.get_collections().collections}
        if COLLECTION not in existing:
            sys.exit(0)

        # Embed query (input_type="query" — optimized for retrieval)
        result = vo.embed(
            [query[:4000]],
            model=VOYAGE_EMBED_MODEL,
            input_type="query",
            truncation=True,
        )
        query_emb = result.embeddings[0]

        # HNSW search in Qdrant
        response = qd.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )
        results = response.points

        if not results:
            sys.exit(0)

        # Output injected into Claude context
        primary_ids = [r.payload['note_id'] for r in results]
        lines = ["=== Relevant vault notes ==="]
        for r in results:
            p = r.payload
            score_pct = int(r.score * 100)
            lines.append(
                f"[[{p['note_id']}]] ({p.get('type', '?')}, {score_pct}%) — {p.get('description', '')}"
            )

        # Graph traversal: BFS 2 levels + backlinks + Qdrant scoring
        outbound, backlinks = load_graph_cache()
        candidate_ids = collect_bfs_candidates(primary_ids, outbound, backlinks)
        scored = score_candidates_qdrant(candidate_ids, query_emb, qd)
        scored_ids = {s["note_id"] for s in scored}
        remaining = MAX_SECONDARY - len(scored)
        if remaining > 0:
            scored.extend(pad_unscored(scored_ids, candidate_ids, remaining))

        if scored:
            lines.append("\n=== Connected notes (graph) ===")
            for c in scored:
                score_str = f", {int(c['score'] * 100)}%" if c.get("score") is not None else ""
                lines.append(f"[[{c['note_id']}]] ({c['type']}{score_str}) — {c['description']}")

        print("\n".join(lines))

        cache_status = "ok" if outbound else "miss"
        log(f"RETRIEVE query={len(query)}c → {len(results)} primary + {len(scored)} graph [cache={cache_status}] (threshold {SCORE_THRESHOLD})")

    except Exception as e:
        log(f"RETRIEVE error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
vault_retrieve.py — UserPromptSubmit hook, active retrieval (v6).

Input stdin : JSON from Claude Code {"prompt": "...", "session_id": "...", ...}
Output      : text injected into Claude context (relevant notes)

v6 improvements:
- Hybrid search: BM25 keyword + vector + Reciprocal Rank Fusion
- Confidence weighting: confirmed notes rank higher
- Temporal decay: recently accessed notes rank higher
- last_retrieved tracking in Qdrant payload
"""

import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date, datetime
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

# Optional config with defaults
try:
    from config import BM25_ENABLED
except ImportError:
    BM25_ENABLED = True
try:
    from config import RRF_K
except ImportError:
    RRF_K = 60
try:
    from config import BM25_TOP_K
except ImportError:
    BM25_TOP_K = 10
try:
    from config import VECTOR_TOP_K
except ImportError:
    VECTOR_TOP_K = 10
try:
    from config import RRF_FINAL_TOP_K
except ImportError:
    RRF_FINAL_TOP_K = 3
try:
    from config import CONFIDENCE_BOOST
except ImportError:
    CONFIDENCE_BOOST = 1.2
try:
    from config import DECAY_ENABLED
except ImportError:
    DECAY_ENABLED = True
try:
    from config import DECAY_HALF_LIFE_DAYS
except ImportError:
    DECAY_HALF_LIFE_DAYS = 90
try:
    from config import DECAY_FLOOR
except ImportError:
    DECAY_FLOOR = 0.3

COLLECTION = "vault_notes"
TODAY = date.today().isoformat()

# Stopwords for BM25 keyword search (common words that add noise)
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "above", "below", "and", "but",
    "or", "not", "no", "if", "then", "than", "so", "that", "this", "it",
    "its", "my", "your", "his", "her", "our", "their", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "each", "every",
    "both", "few", "more", "most", "some", "any", "just", "also", "very",
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "en",
    "que", "qui", "dans", "pour", "sur", "avec", "par", "pas", "plus",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce", "se",
})


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
        data = json.loads(GRAPH_CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("outbound", {}), data.get("backlinks", {})
    except Exception:
        return {}, {}


# ─── BM25 Keyword Search ────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, removing stopwords."""
    words = re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Simple BM25 keyword search over vault notes. Zero external dependencies."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Build mini-index: scan note files
    docs = []
    for p in VAULT_NOTES_DIR.glob("*.md"):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:3000]
            desc_m = re.search(r'^description:\s*(.+)$', text, re.MULTILINE)
            type_m = re.search(r'^type:\s*(.+)$', text, re.MULTILINE)
            conf_m = re.search(r'^confidence:\s*(.+)$', text, re.MULTILINE)
            tokens = tokenize(text)
            docs.append({
                "note_id": p.stem,
                "tokens": tokens,
                "tf": Counter(tokens),
                "len": len(tokens),
                "description": desc_m.group(1).strip() if desc_m else p.stem,
                "type": type_m.group(1).strip() if type_m else "?",
                "confidence": conf_m.group(1).strip() if conf_m else "experimental",
            })
        except Exception:
            continue

    if not docs:
        return []

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    N = len(docs)
    avgdl = sum(d["len"] for d in docs) / N if N else 1

    # Document frequency for each query token
    df = {}
    for token in set(query_tokens):
        df[token] = sum(1 for d in docs if token in d["tf"])

    # Score each document
    for doc in docs:
        score = 0.0
        for token in query_tokens:
            if token not in df or df[token] == 0:
                continue
            idf = math.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1)
            tf = doc["tf"].get(token, 0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc["len"] / avgdl))
            score += idf * tf_norm
        doc["bm25_score"] = score

    # Sort and return top_k
    docs.sort(key=lambda d: d["bm25_score"], reverse=True)
    return [
        {
            "note_id": d["note_id"],
            "description": d["description"],
            "type": d["type"],
            "confidence": d["confidence"],
            "score": d["bm25_score"],
        }
        for d in docs[:top_k]
        if d["bm25_score"] > 0
    ]


# ─── RRF Fusion ─────────────────────────────────────────────────────────────


def rrf_merge(vector_results: list[dict], keyword_results: list[dict], k: int = 60, top_k: int = 3) -> list[dict]:
    """Reciprocal Rank Fusion: merge two ranked lists into one."""
    scores = {}
    metadata = {}

    for rank, item in enumerate(vector_results):
        nid = item["note_id"]
        scores[nid] = scores.get(nid, 0) + 1 / (k + rank + 1)
        metadata[nid] = item

    for rank, item in enumerate(keyword_results):
        nid = item["note_id"]
        scores[nid] = scores.get(nid, 0) + 1 / (k + rank + 1)
        if nid not in metadata:
            metadata[nid] = item

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for nid, fused_score in ranked[:top_k]:
        entry = metadata[nid].copy()
        entry["rrf_score"] = fused_score
        results.append(entry)
    return results


# ─── Decay & Confidence Scoring ─────────────────────────────────────────────


def compute_decay(last_retrieved: str | None, created: str | None) -> float:
    """Compute temporal decay factor based on days since last retrieval."""
    if not DECAY_ENABLED:
        return 1.0

    ref_date = last_retrieved or created or TODAY
    try:
        ref = datetime.strptime(ref_date[:10], "%Y-%m-%d").date()
        days_since = (date.today() - ref).days
    except (ValueError, TypeError):
        days_since = 0

    if days_since <= 0:
        return 1.0

    # Exponential decay: score = e^(-t * ln2 / half_life), floored
    decay = math.exp(-days_since * math.log(2) / DECAY_HALF_LIFE_DAYS)
    return max(DECAY_FLOOR, decay)


def apply_confidence_boost(confidence: str | None) -> float:
    """Return score multiplier based on confidence level."""
    if confidence and confidence.strip().lower() == "confirmed":
        return CONFIDENCE_BOOST
    return 1.0


# ─── Graph Traversal ────────────────────────────────────────────────────────


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
                "confidence": r.payload.get("confidence", "experimental"),
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


# ─── Last Retrieved Tracking ────────────────────────────────────────────────


def update_last_retrieved(note_ids: list[str], qd):
    """Update last_retrieved timestamp for retrieved notes in Qdrant."""
    if not note_ids:
        return
    try:
        import uuid
        from qdrant_client.models import PointStruct
        for nid in note_ids:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, nid))
            qd.set_payload(
                collection_name=COLLECTION,
                payload={"last_retrieved": TODAY},
                points=[point_id],
            )
    except Exception:
        pass  # Non-critical, don't block retrieval


# ─── Main ────────────────────────────────────────────────────────────────────


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

        # ── Vector search ──
        result = vo.embed(
            [query[:4000]],
            model=VOYAGE_EMBED_MODEL,
            input_type="query",
            truncation=True,
        )
        query_emb = result.embeddings[0]

        response = qd.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=VECTOR_TOP_K if BM25_ENABLED else TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )
        vector_results = [
            {
                "note_id": r.payload["note_id"],
                "description": r.payload.get("description", ""),
                "type": r.payload.get("type", "?"),
                "confidence": r.payload.get("confidence", "experimental"),
                "last_retrieved": r.payload.get("last_retrieved"),
                "created": r.payload.get("created"),
                "score": r.score,
            }
            for r in response.points
        ]

        # ── BM25 keyword search + RRF fusion ──
        if BM25_ENABLED and VAULT_NOTES_DIR.exists():
            keyword_results = bm25_search(query, top_k=BM25_TOP_K)
            if keyword_results or vector_results:
                primary = rrf_merge(vector_results, keyword_results, k=RRF_K, top_k=RRF_FINAL_TOP_K)
            else:
                primary = []
        else:
            primary = vector_results[:TOP_K]

        if not primary:
            sys.exit(0)

        # ── Apply decay + confidence scoring to primary results ──
        for note in primary:
            decay = compute_decay(note.get("last_retrieved"), note.get("created"))
            conf_boost = apply_confidence_boost(note.get("confidence"))
            base_score = note.get("rrf_score") or note.get("score") or 0
            note["effective_score"] = base_score * decay * conf_boost

        primary.sort(key=lambda n: n["effective_score"], reverse=True)

        # ── Output primary notes ──
        primary_ids = [n["note_id"] for n in primary]
        lines = ["=== Relevant vault notes ==="]
        for n in primary:
            score_pct = int((n.get("score") or 0) * 100) if n.get("score") else "?"
            conf_tag = " [confirmed]" if n.get("confidence") == "confirmed" else ""
            lines.append(
                f"[[{n['note_id']}]] ({n.get('type', '?')}, {score_pct}%{conf_tag}) — {n.get('description', '')}"
            )

        # ── Graph traversal: BFS 2 levels + backlinks + Qdrant scoring ──
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

        # ── Update last_retrieved for surfaced notes ──
        all_surfaced = primary_ids + [s["note_id"] for s in scored if s.get("score") is not None]
        update_last_retrieved(all_surfaced, qd)

        # ── Logging ──
        cache_status = "ok" if outbound else "miss"
        search_mode = "hybrid" if BM25_ENABLED else "vector"
        log(f"RETRIEVE [{search_mode}] query={len(query)}c → {len(primary)} primary + {len(scored)} graph [cache={cache_status}] (threshold {SCORE_THRESHOLD})")

    except Exception as e:
        log(f"RETRIEVE error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()

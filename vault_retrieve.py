#!/usr/bin/env python3
"""
vault_retrieve.py — UserPromptSubmit hook, active retrieval (v7).

Input stdin : JSON from Claude Code {"prompt": "...", "session_id": "...", ...}
Output      : text injected into Claude context (relevant notes)

v7 improvements (on top of v6):
- Source chunk injection: injects conversation excerpt that generated the note

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
try:
    from config import RERANK_ENABLED
except ImportError:
    RERANK_ENABLED = True
try:
    from config import RERANK_MODEL
except ImportError:
    RERANK_MODEL = "rerank-2"
try:
    from config import RERANK_CANDIDATES
except ImportError:
    RERANK_CANDIDATES = 10
try:
    from config import BM25_INDEX_PATH as _BM25_INDEX_PATH
    BM25_INDEX_PATH = Path(_BM25_INDEX_PATH)
except ImportError:
    BM25_INDEX_PATH = None  # Will fallback to live scan

try:
    from config import SOURCE_CHUNKS_ENABLED
except ImportError:
    SOURCE_CHUNKS_ENABLED = True

try:
    from config import SOURCE_CHUNKS_DIR as _SCD
    SOURCE_CHUNKS_DIR = Path(_SCD)
except ImportError:
    SOURCE_CHUNKS_DIR = VAULT_NOTES_DIR / "_sources"

try:
    from config import SOURCE_INJECT_MAX_CHARS
except ImportError:
    SOURCE_INJECT_MAX_CHARS = 800

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


def _load_bm25_index() -> list[dict] | None:
    """Load persistent BM25 index. Returns None if unavailable."""
    if BM25_INDEX_PATH and BM25_INDEX_PATH.exists():
        try:
            return json.loads(BM25_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _build_live_index() -> list[dict]:
    """Build BM25 index from vault files on the fly (fallback)."""
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
                "tf": dict(Counter(tokens)),
                "len": len(tokens),
                "description": desc_m.group(1).strip() if desc_m else p.stem,
                "type": type_m.group(1).strip() if type_m else "?",
                "confidence": conf_m.group(1).strip() if conf_m else "experimental",
            })
        except Exception:
            continue
    return docs


def _score_bm25(docs: list[dict], query_tokens: list[str]) -> list[dict]:
    """Score documents using BM25 algorithm."""
    k1 = 1.5
    b = 0.75
    N = len(docs)
    avgdl = sum(d["len"] for d in docs) / N if N else 1

    df = {}
    for token in set(query_tokens):
        df[token] = sum(1 for d in docs if token in d.get("tf", {}))

    for doc in docs:
        score = 0.0
        tf_map = doc.get("tf", {})
        for token in query_tokens:
            if token not in df or df[token] == 0:
                continue
            idf = math.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1)
            tf = tf_map.get(token, 0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc["len"] / avgdl))
            score += idf * tf_norm
        doc["bm25_score"] = score

    return docs


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 keyword search. Uses persistent index if available, falls back to live scan."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # Try persistent index first, fall back to live scan
    docs = _load_bm25_index()
    if docs is None:
        docs = _build_live_index()

    if not docs:
        return []

    docs = _score_bm25(docs, query_tokens)
    docs.sort(key=lambda d: d.get("bm25_score", 0), reverse=True)

    return [
        {
            "note_id": d["note_id"],
            "description": d.get("description", d["note_id"]),
            "type": d.get("type", "?"),
            "confidence": d.get("confidence", "experimental"),
            "score": d["bm25_score"],
        }
        for d in docs[:top_k]
        if d.get("bm25_score", 0) > 0
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


# ─── Voyage AI Reranking ────────────────────────────────────────────────────


def rerank_with_voyage(query: str, candidates: list[dict], vo, top_k: int = 3) -> list[dict]:
    """Rerank candidates using Voyage AI reranker for higher precision."""
    try:
        # Build documents list: description + note_id for context
        documents = [
            f"{c.get('description', '')} [{c['note_id']}]"
            for c in candidates
        ]
        result = vo.rerank(
            query=query[:1000],
            documents=documents,
            model=RERANK_MODEL,
            top_k=top_k,
        )
        # Map reranked results back to candidate dicts
        reranked = []
        for r in result.results:
            idx = r.index
            if idx < len(candidates):
                entry = candidates[idx].copy()
                entry["rerank_score"] = r.relevance_score
                reranked.append(entry)
        return reranked
    except Exception as e:
        log(f"RERANK error (falling back to RRF order): {e}")
        return candidates[:top_k]


# ─── Source Chunk Injection ─────────────────────────────────────────────────


def load_source_chunk(note_id: str) -> str | None:
    """Load the source conversation chunk for a note, if available."""
    if not SOURCE_CHUNKS_ENABLED:
        return None
    try:
        chunk_path = SOURCE_CHUNKS_DIR / f"{note_id}.md"
        if not chunk_path.exists():
            return None
        text = chunk_path.read_text(encoding="utf-8", errors="replace")
        # Strip frontmatter from source chunk
        body = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL).strip()
        if body:
            return body[:SOURCE_INJECT_MAX_CHARS]
    except Exception:
        pass
    return None


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
                # Get more candidates for reranking
                rrf_top = RERANK_CANDIDATES if RERANK_ENABLED else RRF_FINAL_TOP_K
                primary = rrf_merge(vector_results, keyword_results, k=RRF_K, top_k=rrf_top)
            else:
                primary = []
        else:
            primary = vector_results[:TOP_K]

        if not primary:
            sys.exit(0)

        # ── Voyage AI reranking (precision layer) ──
        if RERANK_ENABLED and len(primary) > RRF_FINAL_TOP_K:
            primary = rerank_with_voyage(query, primary, vo, top_k=RRF_FINAL_TOP_K)

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

        # ── Source chunk injection (top primary note only, for detail) ──
        if SOURCE_CHUNKS_ENABLED and primary:
            top_note_id = primary[0]["note_id"]
            source = load_source_chunk(top_note_id)
            if source:
                lines.append(f"\n=== Source context for [[{top_note_id}]] ===")
                lines.append(source)

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
        search_mode = "hybrid+rerank" if (BM25_ENABLED and RERANK_ENABLED) else ("hybrid" if BM25_ENABLED else "vector")
        log(f"RETRIEVE [{search_mode}] query={len(query)}c → {len(primary)} primary + {len(scored)} graph [cache={cache_status}] (threshold {SCORE_THRESHOLD})")

    except Exception as e:
        log(f"RETRIEVE error: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()

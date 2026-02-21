from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

from .db import canonical_memory_edge_key

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "this",
    "that",
    "we",
    "you",
    "i",
    "it",
    "de",
    "la",
    "le",
    "les",
    "des",
    "du",
    "et",
    "ou",
    "un",
    "une",
    "en",
    "sur",
    "avec",
    "pas",
    "ne",
    "sans",
}
NEGATIVE_PATTERNS = (
    " not ",
    " never ",
    " no ",
    " pas ",
    " ne ",
    " sans ",
    " interdit ",
    " impossible ",
    " disabled ",
)
DEFAULT_MODEL = os.environ.get("MEMORY_RELATION_MODEL", "claude-sonnet-4-6")


def _normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def tokenize(text: str) -> set[str]:
    base = _normalize_text(text)
    raw = re.findall(r"[a-z0-9][a-z0-9_\-/.:]{1,}", base)
    cleaned: set[str] = set()
    for token in raw:
        token = token.strip("._:-/ ")
        if not token or token in STOPWORDS or len(token) < 2:
            continue
        cleaned.add(token)
    return cleaned


def _is_negative(text: str) -> bool:
    normalized = f" {_normalize_text(text)} "
    return any(tok in normalized for tok in NEGATIVE_PATTERNS)


def _repair_json(raw: str) -> str:
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_str:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = not in_str
            continue
        if in_str and ch in {"\n", "\r", "\t"}:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        out.append(ch)
    return "".join(out)


def _call_llm(prompt: str, timeout_s: int) -> tuple[list[dict[str, Any]], str]:
    if timeout_s <= 0:
        return [], "llm disabled"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", DEFAULT_MODEL],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return [], f"llm call error: {exc}"

    if proc.returncode != 0:
        return [], f"llm rc={proc.returncode}: {proc.stderr[:160]}"

    raw = (proc.stdout or "").strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = _repair_json(raw)
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return [], f"llm json parse error: {exc}"

    pairs = parsed.get("pairs") if isinstance(parsed, dict) else None
    if not isinstance(pairs, list):
        return [], "llm payload missing pairs[]"

    out: list[dict[str, Any]] = []
    for row in pairs:
        if isinstance(row, dict):
            out.append(row)
    return out, ""


def _deterministic_decision(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_tokens: set[str] = a.get("_tokens", set())
    b_tokens: set[str] = b.get("_tokens", set())
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    inter_n = len(inter)
    if inter_n < 2:
        return {"kind": "none", "reason": "low_overlap"}

    a_n = max(1, len(a_tokens))
    b_n = max(1, len(b_tokens))
    cov_small = inter_n / float(min(a_n, b_n))
    jaccard = inter_n / float(max(1, len(union)))
    a_neg = _is_negative(str(a.get("fact_text", "")))
    b_neg = _is_negative(str(b.get("fact_text", "")))

    if a_neg != b_neg and cov_small >= 0.5:
        return {
            "kind": "accepted",
            "relation": "contradicts",
            "confidence": min(0.95, 0.62 + cov_small * 0.3),
            "src_node_id": str(a.get("id")),
            "dst_node_id": str(b.get("id")),
            "decision_source": "deterministic",
            "reason": f"polarity_mismatch cov={cov_small:.2f}",
        }

    subset_ab = len(a_tokens - b_tokens) <= 1 and cov_small >= 0.74
    subset_ba = len(b_tokens - a_tokens) <= 1 and cov_small >= 0.74
    if subset_ab or subset_ba:
        if len(a_tokens) >= len(b_tokens):
            src, dst = str(a.get("id")), str(b.get("id"))
        else:
            src, dst = str(b.get("id")), str(a.get("id"))
        return {
            "kind": "accepted",
            "relation": "updates",
            "confidence": min(0.92, 0.6 + cov_small * 0.35),
            "src_node_id": src,
            "dst_node_id": dst,
            "decision_source": "deterministic",
            "reason": f"subset_like cov={cov_small:.2f}",
        }

    if cov_small >= 0.5 or jaccard >= 0.33:
        return {
            "kind": "accepted",
            "relation": "supports",
            "confidence": min(0.9, 0.5 + cov_small * 0.35),
            "src_node_id": str(a.get("id")),
            "dst_node_id": str(b.get("id")),
            "decision_source": "deterministic",
            "reason": f"supportive_overlap cov={cov_small:.2f} j={jaccard:.2f}",
        }

    if cov_small >= 0.32 or jaccard >= 0.2:
        return {
            "kind": "ambiguous",
            "src_node_id": str(a.get("id")),
            "dst_node_id": str(b.get("id")),
            "reason": f"ambiguous cov={cov_small:.2f} j={jaccard:.2f}",
        }

    return {"kind": "none", "reason": "insufficient_signal"}


def _build_llm_prompt(items: list[dict[str, Any]]) -> str:
    payload_lines = []
    for item in items:
        payload_lines.append(
            f"- pair_id: {item['pair_id']}\n"
            f"  text_a: {item['text_a']}\n"
            f"  text_b: {item['text_b']}"
        )
    joined = "\n".join(payload_lines)
    return f"""Classify semantic relation for each pair.
Return ONLY JSON object with key 'pairs'.
Each output item: {{"pair_id":"...","relation":"supports|updates|contradicts|none","confidence":0..1,"direction":"a_to_b|b_to_a"}}.
Use direction only for updates; otherwise keep a_to_b.

Pairs:
{joined}
"""


def generate_relation_candidates(nodes: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    max_pairs = max(1, int(getattr(settings, "relation_batch_max_pairs", 800)))
    min_conf = max(0.0, min(1.0, float(getattr(settings, "relation_min_confidence", 0.72))))
    llm_timeout = int(getattr(settings, "relation_llm_timeout", 10))

    prepared: list[dict[str, Any]] = []
    for raw in nodes:
        node_id = str(raw.get("id", "")).strip()
        fact_text = str(raw.get("fact_text", "")).strip()
        if not node_id or not fact_text:
            continue
        tokens = tokenize(fact_text)
        if len(tokens) < 2:
            continue
        row = dict(raw)
        row["_tokens"] = tokens
        prepared.append(row)

    token_index: dict[str, list[int]] = {}
    for idx, row in enumerate(prepared):
        for token in row["_tokens"]:
            token_index.setdefault(token, []).append(idx)

    pair_shared: dict[tuple[int, int], int] = {}
    for postings in token_index.values():
        if len(postings) < 2 or len(postings) > 60:
            continue
        for i in range(len(postings)):
            for j in range(i + 1, len(postings)):
                a, b = postings[i], postings[j]
                if a > b:
                    a, b = b, a
                pair_shared[(a, b)] = pair_shared.get((a, b), 0) + 1

    ranked_pairs = sorted(pair_shared.items(), key=lambda kv: kv[1], reverse=True)
    ranked_pairs = ranked_pairs[: max_pairs * 2]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for pair_rank, ((i, j), _shared) in enumerate(ranked_pairs, start=1):
        a = prepared[i]
        b = prepared[j]
        decision = _deterministic_decision(a, b)
        kind = decision.get("kind")
        if kind == "accepted":
            relation = str(decision["relation"])
            src = str(decision["src_node_id"])
            dst = str(decision["dst_node_id"])
            conf = float(decision["confidence"])
            if src == dst:
                continue
            key = canonical_memory_edge_key(src, dst, relation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row = {
                "src_node_id": src,
                "dst_node_id": dst,
                "relation": relation,
                "confidence": round(conf, 4),
                "decision_source": str(decision.get("decision_source", "deterministic")),
                "reason": str(decision.get("reason", "")),
                "canonical_key": key,
                "status": "accepted" if conf >= min_conf else "rejected",
            }
            if row["status"] == "accepted":
                accepted.append(row)
            else:
                rejected.append(row)
            continue

        if kind == "ambiguous":
            ambiguous.append(
                {
                    "pair_id": f"p{pair_rank}",
                    "a_id": str(a.get("id")),
                    "b_id": str(b.get("id")),
                    "text_a": str(a.get("fact_text", ""))[:260],
                    "text_b": str(b.get("fact_text", ""))[:260],
                    "reason": str(decision.get("reason", "")),
                }
            )

    llm_error = ""
    llm_rows: list[dict[str, Any]] = []
    if ambiguous:
        batch = ambiguous[: min(len(ambiguous), max_pairs)]
        llm_rows, llm_error = _call_llm(_build_llm_prompt(batch), llm_timeout)
        by_pair = {str(item["pair_id"]): item for item in batch}
        for row in llm_rows:
            pair_id = str(row.get("pair_id", "")).strip()
            if not pair_id or pair_id not in by_pair:
                continue
            relation = str(row.get("relation", "none")).strip().lower()
            if relation not in {"supports", "updates", "contradicts"}:
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            direction = str(row.get("direction", "a_to_b")).strip().lower()
            pair = by_pair[pair_id]
            src = pair["a_id"]
            dst = pair["b_id"]
            if relation == "updates" and direction == "b_to_a":
                src, dst = dst, src
            if src == dst:
                continue
            key = canonical_memory_edge_key(src, dst, relation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidate = {
                "src_node_id": src,
                "dst_node_id": dst,
                "relation": relation,
                "confidence": round(confidence, 4),
                "decision_source": "llm",
                "reason": pair.get("reason", "ambiguous_resolved"),
                "canonical_key": key,
                "status": "accepted" if confidence >= min_conf else "rejected",
            }
            if candidate["status"] == "accepted":
                accepted.append(candidate)
            else:
                rejected.append(candidate)

    accepted = accepted[: max_pairs]
    rejected = rejected[: max_pairs]
    return {
        "accepted": accepted,
        "rejected": rejected,
        "stats": {
            "nodes_considered": len(prepared),
            "pairs_scored": len(pair_shared),
            "pairs_ranked": len(ranked_pairs),
            "ambiguous": len(ambiguous),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "llm_rows": len(llm_rows),
        },
        "llm_error": llm_error,
    }

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any


DEFAULT_MODEL = os.environ.get("MEMORY_LIVE_EXTRACT_MODEL", "claude-sonnet-4-6")
ALLOWED_TYPES = {"decision", "preference", "constraint", "fact"}


def _as_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USE_CLAUDE = _as_bool("MEMORY_USE_CLAUDE_LIVE_EXTRACT", True)


def _repair_json_newlines(raw: str) -> str:
    result = []
    in_string = False
    escaped = False
    for char in raw:
        if escaped:
            result.append(char)
            escaped = False
        elif char == "\\" and in_string:
            result.append(char)
            escaped = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == "\n" and in_string:
            result.append("\\n")
        elif char == "\r" and in_string:
            result.append("\\r")
        elif char == "\t" and in_string:
            result.append("\\t")
        else:
            result.append(char)
    return "".join(result)


def _call_claude(prompt: str, timeout_s: int) -> str:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    result = subprocess.run(
        ["claude", "-p", prompt, "--model", DEFAULT_MODEL],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    if result.returncode != 0:
        details = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(f"claude -p exit {result.returncode}: {details[:240]}")
    text = (result.stdout or "").strip()
    if not text:
        raise RuntimeError("claude -p empty response")
    return text


def _build_prompt(payload: dict[str, Any]) -> str:
    turns = payload.get("turns", [])
    max_candidates = int(payload.get("max_candidates", 3))
    conversation = []
    for turn in turns[-12:]:
        role = str(turn.get("role", "user")).upper()
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        conversation.append(f"{role}: {text[:1000]}")
    convo = "\n\n".join(conversation)

    return f"""You are a strict JSON extraction bot.
Task: extract only durable facts from this short active-session context.

RULES:
- Return ONLY JSON object: {{"candidates":[...]}}
- Max {max_candidates} candidates.
- Candidate fields:
  - fact_text: short proposition, max 220 chars
  - fact_type: one of decision|preference|constraint|fact
  - confidence: float [0,1]
  - evidence_excerpt: quote/paraphrase from transcript, max 220 chars
- Ignore temporary debugging and uncertain speculation.
- If no durable candidates: {{"candidates":[]}}

TRANSCRIPT:
{convo}
"""


def _fallback_extract(payload: dict[str, Any]) -> dict[str, Any]:
    turns = payload.get("turns", [])
    if not turns:
        return {"candidates": []}

    max_candidates = int(payload.get("max_candidates", 3))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _candidate_for_text(text: str) -> tuple[str, float] | None:
        # Conservative heuristic fallback when claude -p is unavailable.
        if re.search(r"\b(on garde|décision finale|c[' ]?est validé|this is final)\b", text, flags=re.IGNORECASE):
            return ("decision", 0.62)
        if re.search(r"\b(je pr[ée]f[èe]re|on pr[ée]f[èe]re|prefer|par d[ée]faut|default)\b", text, flags=re.IGNORECASE):
            return ("preference", 0.46)
        if re.search(r"\b(il faut|doit|must|sans |ne pas|pas de|interdit|only)\b", text, flags=re.IGNORECASE):
            return ("constraint", 0.44)
        if re.search(r"\b(on utilise|utiliser|stack|architecture|workflow)\b", text, flags=re.IGNORECASE):
            return ("fact", 0.36)
        return None

    for turn in reversed(turns[-12:]):
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        inferred = _candidate_for_text(text)
        if not inferred:
            continue
        fact_type, confidence = inferred
        fact_text = text[:220]
        key = re.sub(r"\s+", " ", fact_text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "fact_text": fact_text,
                "fact_type": fact_type,
                "confidence": confidence,
                "evidence_excerpt": fact_text,
            }
        )
        if len(candidates) >= max_candidates:
            break

    return {"candidates": candidates}


def _normalize(parsed: dict[str, Any], max_candidates: int) -> dict[str, Any]:
    raw_candidates = parsed.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    out = []
    for item in raw_candidates[:max_candidates]:
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact_text", "")).strip()
        if not fact_text:
            continue
        fact_type = str(item.get("fact_type", "fact")).strip().lower()
        if fact_type not in ALLOWED_TYPES:
            fact_type = "fact"
        try:
            confidence = float(item.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        evidence_excerpt = str(item.get("evidence_excerpt", "")).strip()
        out.append(
            {
                "fact_text": fact_text[:220],
                "fact_type": fact_type,
                "confidence": confidence,
                "evidence_excerpt": evidence_excerpt[:220] or fact_text[:220],
            }
        )
    return {"candidates": out}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}

    max_candidates = int(payload.get("max_candidates", 3))
    timeout_s = int(payload.get("timeout_s", 8))
    prompt = _build_prompt(payload)

    if USE_CLAUDE:
        try:
            raw = _call_claude(prompt, timeout_s=timeout_s)
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = _repair_json_newlines(raw)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {"candidates": []}
            result = _normalize(parsed, max_candidates=max_candidates)
        except Exception as exc:
            result = _fallback_extract(payload)
            result["error"] = str(exc)
    else:
        result = _fallback_extract(payload)

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

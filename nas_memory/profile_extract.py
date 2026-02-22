from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any


DEFAULT_MODEL = os.environ.get("MEMORY_PROFILE_EXTRACT_MODEL", "claude-sonnet-4-6")
ALLOWED_TYPES = {"decision", "preference", "constraint", "fact"}


def _as_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


USE_CLAUDE = _as_bool("MEMORY_USE_CLAUDE_PROFILE_EXTRACT", True)


def _repair_json(raw: str) -> str:
    out = []
    in_str = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\" and in_str:
            out.append(ch)
            escaped = True
        elif ch == '"':
            out.append(ch)
            in_str = not in_str
        elif ch == "\n" and in_str:
            out.append("\\n")
        elif ch == "\r" and in_str:
            out.append("\\r")
        elif ch == "\t" and in_str:
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


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


def _normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    fact_text = str(item.get("fact_text", "")).strip()
    if not fact_text:
        return None
    fact_type = str(item.get("fact_type", "fact")).strip().lower()
    if fact_type not in ALLOWED_TYPES:
        fact_type = "fact"
    try:
        confidence = float(item.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "fact_text": fact_text[:220],
        "fact_type": fact_type,
        "confidence": confidence,
    }


def _normalize_payload(payload: dict[str, Any], max_items: int) -> dict[str, Any]:
    out = {"static": [], "dynamic": []}
    for bucket in ("static", "dynamic"):
        raw = payload.get(bucket, [])
        if not isinstance(raw, list):
            raw = []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in raw[:max_items]:
            if not isinstance(entry, dict):
                continue
            item = _normalize_item(entry)
            if not item:
                continue
            key = f"{item['fact_type']}|{item['fact_text'].lower()}"
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
        out[bucket] = normalized[:max_items]
    return out


def _fallback_extract(payload: dict[str, Any], max_items: int) -> dict[str, Any]:
    text = str(payload.get("conversation_text", "")).strip()
    staging = payload.get("staging_items", [])
    if not isinstance(staging, list):
        staging = []

    static: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    seen: set[str] = set()

    patterns = [
        (r"\b(on garde|d[ée]cision finale|c[' ]?est valid[ée]|approved|confirmed)\b", "decision", 0.62, static),
        (r"\b(pr[ée]f[èe]re|prefer|par d[ée]faut|default)\b", "preference", 0.52, static),
        (r"\b(il faut|doit|must|sans |ne pas|pas de|interdit|only)\b", "constraint", 0.54, static),
        (r"\b(on utilise|stack|architecture|workflow)\b", "fact", 0.40, dynamic),
    ]

    for line in text.splitlines()[-120:]:
        candidate = line.strip()
        if not candidate:
            continue
        for pattern, fact_type, conf, bucket in patterns:
            if re.search(pattern, candidate, flags=re.IGNORECASE):
                key = f"{fact_type}|{candidate.lower()[:220]}"
                if key in seen:
                    break
                seen.add(key)
                bucket.append(
                    {
                        "fact_text": candidate[:220],
                        "fact_type": fact_type,
                        "confidence": conf,
                    }
                )
                break

    for item in staging:
        if len(dynamic) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact_text", "")).strip()
        fact_type = str(item.get("fact_type", "fact")).strip().lower()
        if not fact_text:
            continue
        if fact_type not in ALLOWED_TYPES:
            fact_type = "fact"
        key = f"{fact_type}|{fact_text.lower()[:220]}"
        if key in seen:
            continue
        seen.add(key)
        dynamic.append(
            {
                "fact_text": fact_text[:220],
                "fact_type": fact_type,
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.35) or 0.35))),
            }
        )

    return {"static": static[:max_items], "dynamic": dynamic[:max_items]}


def _build_prompt(payload: dict[str, Any], max_items: int) -> str:
    conversation = str(payload.get("conversation_text", ""))[:12000]
    staging = payload.get("staging_items", [])
    if not isinstance(staging, list):
        staging = []
    staging_lines = []
    for item in staging[:20]:
        if not isinstance(item, dict):
            continue
        staging_lines.append(
            f"- {item.get('fact_type', 'fact')} ({item.get('status', 'experimental')}): {str(item.get('fact_text', ''))[:180]}"
        )
    staging_text = "\n".join(staging_lines)
    return f"""You extract high-level memory profile from a session.
Return ONLY JSON object with keys "static" and "dynamic".
Each list item has fields: fact_text, fact_type(decision|preference|constraint|fact), confidence [0..1].
Max {max_items} items per list.

Rules:
- static: stable preferences/constraints/decisions expected to hold across sessions.
- dynamic: short-lived current-context facts useful for near-term tasks.
- avoid duplicates and speculation.

SESSION TRANSCRIPT:
{conversation}

STAGING CANDIDATES:
{staging_text}
"""


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    max_items = int(payload.get("max_items", 12))
    timeout_s = int(payload.get("timeout_s", 12))
    prompt = _build_prompt(payload, max_items=max_items)

    if USE_CLAUDE:
        try:
            raw = _call_claude(prompt, timeout_s=timeout_s)
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = _repair_json(raw)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {"static": [], "dynamic": []}
            result = _normalize_payload(parsed, max_items=max_items)
        except Exception as exc:
            result = _fallback_extract(payload, max_items=max_items)
            result["error"] = str(exc)
    else:
        result = _fallback_extract(payload, max_items=max_items)

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

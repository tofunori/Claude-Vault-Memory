#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    env = _load_env(Path("/volume1/Services/memory/.env"))
    api_url = os.environ.get("MEMORY_API_URL") or env.get("MEMORY_API_URL") or "http://127.0.0.1:8876"
    token = os.environ.get("MEMORY_API_TOKEN") or env.get("MEMORY_API_TOKEN") or ""
    if not token:
        print("ERROR: MEMORY_API_TOKEN missing")
        return 2

    payload = {"actor": "relation_compact_job"}
    req = request.Request(
        f"{api_url.rstrip('/')}/admin/relations/compact",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(body)
            return 0
    except error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

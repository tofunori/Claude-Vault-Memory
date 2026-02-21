#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from nas_memory.burnin.report import STRICT_THRESHOLDS, generate_report
    from nas_memory.burnin.scenarios import run_synthetic_cycle
except Exception:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nas_memory.burnin.report import STRICT_THRESHOLDS, generate_report  # type: ignore
    from nas_memory.burnin.scenarios import run_synthetic_cycle  # type: ignore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_env_file(path: Path) -> dict[str, str]:
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


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ApiClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 12.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
                return {
                    "ok": (resp.status < 400) and isinstance(data, dict) and data.get("status") != "error",
                    "http_status": resp.status,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "data": data,
                    "error": "",
                }
        except error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {}
            return {
                "ok": False,
                "http_status": exc.code,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "data": data,
                "error": f"HTTP {exc.code}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "http_status": None,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "data": {},
                "error": str(exc),
            }

    def get(self, path: str) -> dict[str, Any]:
        req = request.Request(
            f"{self.base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        t0 = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw else {}
                return {
                    "ok": (resp.status < 400) and isinstance(data, dict) and data.get("status") != "error",
                    "http_status": resp.status,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "data": data,
                    "error": "",
                }
        except error.HTTPError as exc:
            return {
                "ok": False,
                "http_status": exc.code,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "data": {},
                "error": f"HTTP {exc.code}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "http_status": None,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "data": {},
                "error": str(exc),
            }


def _micros_to_iso(v: str) -> str:
    try:
        micros = int(v)
        return datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return utc_now_iso()


def _collect_worker_live_samples(
    *,
    unit: str,
    since_micros: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    since_secs = max(0, int((since_micros - 1_000_000) / 1_000_000))
    cmd = [
        "journalctl",
        "--user",
        "-u",
        unit,
        "--no-pager",
        "-o",
        "json",
        "--since",
        f"@{since_secs}",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return since_micros, [], []
    if proc.returncode != 0:
        return since_micros, [], []

    out: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    next_micros = since_micros
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
        except Exception:
            continue

        try:
            row_micros = int(envelope.get("__REALTIME_TIMESTAMP", "0"))
        except Exception:
            row_micros = 0
        if row_micros <= since_micros:
            continue
        next_micros = max(next_micros, row_micros)

        raw_message = str(envelope.get("MESSAGE", "") or "").strip()
        if not raw_message.startswith("{"):
            continue
        try:
            msg = json.loads(raw_message)
        except Exception:
            continue

        if msg.get("message") == "memory_action":
            actions.append(
                {
                    "ts": _micros_to_iso(str(row_micros)),
                    "action": str(msg.get("action", "")),
                    "global_key": str(msg.get("global_key", "")),
                    "relation": str(msg.get("relation", "")),
                    "from_version": msg.get("from_version"),
                    "to_version": msg.get("to_version"),
                }
            )
            continue

        if msg.get("message") != "turn live processed":
            continue

        out.append(
            {
                "ts": _micros_to_iso(str(row_micros)),
                "event_id": msg.get("event_id", ""),
                "session_id": msg.get("session_id", ""),
                "agent": msg.get("agent", ""),
                "turn_index": int(msg.get("turn_index", 0) or 0),
                "candidates": int(msg.get("candidates", 0) or 0),
                "promoted": int(msg.get("promoted", 0) or 0),
                "error": str(msg.get("error", "") or ""),
            }
        )
    return next_micros, out, actions


def _sample_health(client: ApiClient) -> dict[str, Any]:
    ts = utc_now_iso()
    resp = client.get("/health")
    data = resp.get("data") or {}
    queue_db = data.get("queue_db") or {}
    counts = queue_db.get("counts") or {}
    return {
        "ts": ts,
        "ok": bool(resp.get("ok")),
        "http_status": resp.get("http_status"),
        "latency_ms": int(resp.get("latency_ms", 0) or 0),
        "api_status": data.get("status", "error"),
        "queue_counts": {
            "queued": int(counts.get("queued", 0) or 0),
            "processing": int(counts.get("processing", 0) or 0),
            "done": int(counts.get("done", 0) or 0),
            "error": int(counts.get("error", 0) or 0),
        },
        "oldest_queued_at": queue_db.get("oldest_queued_at"),
        "error": str(resp.get("error", "") or ""),
    }


def _sample_retrieve(client: ApiClient, run_id: str) -> dict[str, Any]:
    ts = utc_now_iso()
    resp = client.post(
        "/retrieve",
        {
            "query": "Rappel: quelle architecture mémoire NAS est en place?",
            "session_id": f"burnin-probe-{run_id}",
            "agent": "codex",
            "top_k": 5,
        },
    )
    data = resp.get("data") or {}
    diagnostics = data.get("diagnostics") or {}
    return {
        "ts": ts,
        "ok": bool(resp.get("ok")),
        "http_status": resp.get("http_status"),
        "latency_ms": int(resp.get("latency_ms", 0) or 0),
        "api_status": data.get("status", "error"),
        "diagnostic_latency_ms": int(diagnostics.get("latency_ms", 0) or 0),
        "search_mode": diagnostics.get("search_mode", ""),
        "experimental_used": int(diagnostics.get("experimental_used", 0) or 0),
        "error": str(resp.get("error", "") or data.get("message", "") or ""),
    }


def _sample_graph(client: ApiClient) -> dict[str, Any]:
    ts = utc_now_iso()
    resp = client.get("/admin/graph")
    data = resp.get("data") or {}
    stats = data.get("stats") or {}
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []

    memory_nodes = [n for n in nodes if isinstance(n, dict) and n.get("kind") == "memory" and n.get("status") == "active"]
    memory_ids = {str(n.get("id", "")) for n in memory_nodes if n.get("id")}
    degrees = {node_id: 0 for node_id in memory_ids}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source", ""))
        dst = str(edge.get("target", ""))
        if src in degrees:
            degrees[src] += 1
        if dst in degrees:
            degrees[dst] += 1
    singleton_count = sum(1 for v in degrees.values() if v == 0)

    return {
        "ts": ts,
        "ok": bool(resp.get("ok")),
        "http_status": resp.get("http_status"),
        "latency_ms": int(resp.get("latency_ms", 0) or 0),
        "graph_status": data.get("status", "error"),
        "note_nodes": int(stats.get("note_nodes", 0) or 0),
        "memory_nodes": int(stats.get("memory_nodes", len(memory_nodes)) or 0),
        "memory_edges": int(stats.get("memory_edges", 0) or 0),
        "bridge_edges": int(stats.get("bridge_edges", 0) or 0),
        "components": int(stats.get("components", 0) or 0),
        "memory_singletons": int(singleton_count),
        "error": str(resp.get("error", "") or ""),
    }


def _sample_relation_stats(client: ApiClient) -> dict[str, Any]:
    ts = utc_now_iso()
    resp = client.get("/admin/relations/stats")
    data = resp.get("data") or {}
    stats = data.get("stats") if isinstance(data, dict) else {}
    if not isinstance(stats, dict):
        stats = {}
    candidate_counts = stats.get("candidate_counts") if isinstance(stats.get("candidate_counts"), dict) else {}
    source_counts = stats.get("source_counts") if isinstance(stats.get("source_counts"), dict) else {}
    return {
        "ts": ts,
        "ok": bool(resp.get("ok")),
        "http_status": resp.get("http_status"),
        "latency_ms": int(resp.get("latency_ms", 0) or 0),
        "api_status": data.get("status", "error"),
        "memory_edges_total": int(stats.get("memory_edges_total", 0) or 0),
        "memory_nodes_active": int(stats.get("memory_nodes_active", 0) or 0),
        "memory_singletons": int(stats.get("memory_singletons", 0) or 0),
        "candidate_counts": candidate_counts,
        "source_counts": source_counts,
        "last_candidate_at": stats.get("last_candidate_at"),
        "error": str(resp.get("error", "") or data.get("message", "") or ""),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NAS memory burn-in collector.")
    parser.add_argument("--duration-hours", type=float, default=72.0)
    parser.add_argument("--mode", choices=["mixed", "real", "synthetic"], default="mixed")
    parser.add_argument("--gate", choices=["strict"], default="strict")
    parser.add_argument("--api-url", default=os.getenv("MEMORY_API_URL", "http://192.168.40.2:8876"))
    parser.add_argument("--api-token", default=os.getenv("MEMORY_API_TOKEN", ""))
    parser.add_argument("--out-root", default="/volume1/Services/memory/state/burnin")
    parser.add_argument("--env-file", default="/volume1/Services/memory/.env")
    parser.add_argument("--health-interval", type=int, default=30)
    parser.add_argument("--retrieve-interval", type=int, default=120)
    parser.add_argument("--scenario-interval", type=int, default=300)
    parser.add_argument("--graph-interval", type=int, default=300)
    parser.add_argument("--journal-unit", default="memory-worker.service")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env_file = Path(args.env_file)
    env_map = _read_env_file(env_file)
    api_url = args.api_url or env_map.get("MEMORY_API_URL", "http://192.168.40.2:8876")
    api_token = args.api_token or env_map.get("MEMORY_API_TOKEN", "")
    if not api_token:
        print("ERROR: missing MEMORY_API_TOKEN (arg/env/.env)")
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "duration_hours": args.duration_hours,
        "mode": args.mode,
        "gate": args.gate,
        "api_url": api_url,
        "health_interval_s": args.health_interval,
        "retrieve_interval_s": args.retrieve_interval,
        "scenario_interval_s": args.scenario_interval,
        "graph_interval_s": args.graph_interval,
        "journal_unit": args.journal_unit,
        "thresholds": STRICT_THRESHOLDS,
    }
    _json_write(run_dir / "config.json", config)
    for name in (
        "health_samples.jsonl",
        "retrieve_samples.jsonl",
        "worker_live_samples.jsonl",
        "memory_action_samples.jsonl",
        "synthetic_trace.jsonl",
        "graph_samples.jsonl",
        "relation_stats_samples.jsonl",
    ):
        (run_dir / name).touch(exist_ok=True)

    client = ApiClient(api_url, api_token, timeout_s=args.timeout_s)

    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        print(f"[burnin] stop requested signal={signum}")
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    started = time.monotonic()
    deadline = started + max(1.0, args.duration_hours * 3600.0)
    next_health = started
    next_retrieve = started
    next_scenario = started
    next_graph = started
    worker_since_micros = int(time.time() * 1_000_000)
    cycle_index = 0

    print(f"[burnin] run_dir={run_dir}")
    print(f"[burnin] mode={args.mode} gate={args.gate} duration_h={args.duration_hours}")

    while not stop and time.monotonic() < deadline:
        now = time.monotonic()

        if now >= next_health:
            health = _sample_health(client)
            _append_jsonl(run_dir / "health_samples.jsonl", health)
            worker_since_micros, live_rows, action_rows = _collect_worker_live_samples(
                unit=args.journal_unit,
                since_micros=worker_since_micros,
            )
            for row in live_rows:
                _append_jsonl(run_dir / "worker_live_samples.jsonl", row)
            for row in action_rows:
                _append_jsonl(run_dir / "memory_action_samples.jsonl", row)
            next_health = now + args.health_interval

        if args.mode in {"mixed", "synthetic"} and now >= next_retrieve:
            retrieve = _sample_retrieve(client, run_id)
            _append_jsonl(run_dir / "retrieve_samples.jsonl", retrieve)
            next_retrieve = now + args.retrieve_interval

        if args.mode in {"mixed", "synthetic"} and now >= next_scenario:
            cycle_index += 1
            trace = run_synthetic_cycle(client, run_id=run_id, cycle_index=cycle_index)
            _append_jsonl(run_dir / "synthetic_trace.jsonl", trace)
            next_scenario = now + args.scenario_interval

        if now >= next_graph:
            graph_sample = _sample_graph(client)
            _append_jsonl(run_dir / "graph_samples.jsonl", graph_sample)
            relation_sample = _sample_relation_stats(client)
            _append_jsonl(run_dir / "relation_stats_samples.jsonl", relation_sample)
            next_graph = now + args.graph_interval

        time.sleep(max(0.1, args.sleep))

    # Final worker log sweep before report.
    worker_since_micros, live_rows, action_rows = _collect_worker_live_samples(
        unit=args.journal_unit,
        since_micros=worker_since_micros,
    )
    for row in live_rows:
        _append_jsonl(run_dir / "worker_live_samples.jsonl", row)
    for row in action_rows:
        _append_jsonl(run_dir / "memory_action_samples.jsonl", row)

    config["finished_at"] = utc_now_iso()
    config["duration_seconds_effective"] = int(max(0.0, time.monotonic() - started))
    _json_write(run_dir / "config.json", config)

    summary, passfail = generate_report(run_dir, gate=args.gate)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "overall_pass": bool(passfail.get("overall_pass")),
                "metrics": summary.get("metrics", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passfail.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())

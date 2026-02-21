#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRICT_THRESHOLDS: dict[str, float] = {
    "queue_error_rate_max": 0.005,
    "queue_backlog_stall_minutes_max": 10.0,
    "retrieve_p95_ms_max": 500.0,
    "retrieve_p99_ms_max": 1200.0,
    "retrieve_error_rate_max": 0.01,
    "promotion_rate_min": 0.20,
    "noise_rate_max": 0.80,
    "session_leak_count_max": 0.0,
    "memory_singleton_reduction_min": 0.40,
    "relation_precision_sample_min": 0.85,
}


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        except Exception:
            continue
    return rows


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_queue_metrics(health_samples: list[dict[str, Any]]) -> dict[str, float]:
    if not health_samples:
        return {
            "queue_error_rate": 0.0,
            "queue_backlog_stall_minutes": 0.0,
            "queue_backlog_stall_total_minutes": 0.0,
            "done_delta": 0.0,
            "error_delta": 0.0,
        }

    first = health_samples[0].get("queue_counts", {})
    last = health_samples[-1].get("queue_counts", {})
    done_delta = float(last.get("done", 0) - first.get("done", 0))
    error_delta = float(last.get("error", 0) - first.get("error", 0))
    denom = max(done_delta + error_delta, 1.0)
    queue_error_rate = error_delta / denom

    max_contiguous_s = 0.0
    total_stall_s = 0.0
    active_start: datetime | None = None
    last_ts: datetime | None = None

    for sample in health_samples:
        sample_ts = _parse_iso(str(sample.get("ts", "")))
        if sample_ts is None:
            continue
        queued = int(sample.get("queue_counts", {}).get("queued", 0))
        oldest_queued_at = _parse_iso(str(sample.get("oldest_queued_at", "")))
        is_stalled = False
        if queued > 0 and oldest_queued_at is not None:
            age_s = (sample_ts - oldest_queued_at).total_seconds()
            is_stalled = age_s > 300.0

        if is_stalled:
            if active_start is None:
                active_start = sample_ts
        else:
            if active_start is not None and last_ts is not None:
                span = max(0.0, (last_ts - active_start).total_seconds())
                max_contiguous_s = max(max_contiguous_s, span)
                total_stall_s += span
            active_start = None

        last_ts = sample_ts

    if active_start is not None and last_ts is not None:
        span = max(0.0, (last_ts - active_start).total_seconds())
        max_contiguous_s = max(max_contiguous_s, span)
        total_stall_s += span

    return {
        "queue_error_rate": round(queue_error_rate, 6),
        "queue_backlog_stall_minutes": round(max_contiguous_s / 60.0, 3),
        "queue_backlog_stall_total_minutes": round(total_stall_s / 60.0, 3),
        "done_delta": done_delta,
        "error_delta": error_delta,
    }


def compute_retrieve_metrics(retrieve_samples: list[dict[str, Any]]) -> dict[str, float]:
    total = len(retrieve_samples)
    if total == 0:
        return {
            "retrieve_total": 0.0,
            "retrieve_error_rate": 0.0,
            "retrieve_p50_ms": 0.0,
            "retrieve_p95_ms": 0.0,
            "retrieve_p99_ms": 0.0,
        }

    errors = 0
    latencies: list[float] = []
    for sample in retrieve_samples:
        ok = bool(sample.get("ok", False))
        latency_ms = float(sample.get("latency_ms", 0.0) or 0.0)
        if ok:
            latencies.append(latency_ms)
        else:
            errors += 1

    return {
        "retrieve_total": float(total),
        "retrieve_error_rate": round(errors / max(total, 1), 6),
        "retrieve_p50_ms": round(percentile(latencies, 50), 3),
        "retrieve_p95_ms": round(percentile(latencies, 95), 3),
        "retrieve_p99_ms": round(percentile(latencies, 99), 3),
    }


def compute_live_metrics(worker_live_samples: list[dict[str, Any]]) -> dict[str, float]:
    total = len(worker_live_samples)
    if total == 0:
        return {
            "live_samples_total": 0.0,
            "promotion_rate": 0.0,
            "noise_rate": 1.0,
            "live_extract_error_rate": 0.0,
            "sum_candidates": 0.0,
            "sum_promoted": 0.0,
        }

    sum_candidates = 0
    sum_promoted = 0
    errors = 0
    for sample in worker_live_samples:
        candidates = int(sample.get("candidates", 0) or 0)
        promoted = int(sample.get("promoted", 0) or 0)
        err = str(sample.get("error", "") or "").strip()
        sum_candidates += max(0, candidates)
        sum_promoted += max(0, promoted)
        if err:
            errors += 1

    promotion_rate = float(sum_promoted) / float(max(sum_candidates, 1))
    noise_rate = 1.0 - promotion_rate
    return {
        "live_samples_total": float(total),
        "promotion_rate": round(promotion_rate, 6),
        "noise_rate": round(noise_rate, 6),
        "live_extract_error_rate": round(errors / max(total, 1), 6),
        "sum_candidates": float(sum_candidates),
        "sum_promoted": float(sum_promoted),
    }


def compute_synthetic_metrics(synthetic_trace: list[dict[str, Any]]) -> dict[str, float]:
    if not synthetic_trace:
        return {
            "synthetic_cycles": 0.0,
            "synthetic_success_rate": 0.0,
            "session_leak_count": 0.0,
        }

    total = len(synthetic_trace)
    success = 0
    leaks = 0
    for row in synthetic_trace:
        if bool(row.get("success", False)):
            success += 1
        if not bool(row.get("isolation_ok", True)):
            leaks += 1
    return {
        "synthetic_cycles": float(total),
        "synthetic_success_rate": round(success / max(total, 1), 6),
        "session_leak_count": float(leaks),
    }


def compute_action_metrics(action_samples: list[dict[str, Any]]) -> dict[str, float]:
    if not action_samples:
        return {
            "profile_update_rate": 0.0,
            "forget_actions_count": 0.0,
            "contradiction_resolution_count": 0.0,
            "memory_actions_total": 0.0,
        }
    total = len(action_samples)
    compact_count = 0
    forget_count = 0
    contradict_count = 0
    for row in action_samples:
        action = str(row.get("action", "")).strip()
        relation = str(row.get("relation", "")).strip()
        if action in {"upsert", "admin_upsert"}:
            compact_count += 1
        if action == "forget":
            forget_count += 1
        if relation == "contradicts":
            contradict_count += 1
    return {
        "profile_update_rate": round(compact_count / max(total, 1), 6),
        "forget_actions_count": float(forget_count),
        "contradiction_resolution_count": float(contradict_count),
        "memory_actions_total": float(total),
    }


def compute_graph_metrics(graph_samples: list[dict[str, Any]]) -> dict[str, float]:
    if not graph_samples:
        return {
            "memory_singletons_baseline": 0.0,
            "memory_singletons_now": 0.0,
            "memory_singleton_reduction": 0.0,
            "graph_memory_nodes_baseline": 0.0,
            "graph_memory_nodes_now": 0.0,
        }

    valid = [row for row in graph_samples if bool(row.get("ok"))]
    rows = valid if valid else graph_samples
    baseline = rows[0]
    current = rows[-1]
    baseline_singletons = float(baseline.get("memory_singletons", 0) or 0)
    current_singletons = float(current.get("memory_singletons", 0) or 0)
    if baseline_singletons > 0:
        reduction = 1.0 - (current_singletons / baseline_singletons)
    else:
        reduction = 0.0
    return {
        "memory_singletons_baseline": baseline_singletons,
        "memory_singletons_now": current_singletons,
        "memory_singleton_reduction": round(reduction, 6),
        "graph_memory_nodes_baseline": float(baseline.get("memory_nodes", 0) or 0),
        "graph_memory_nodes_now": float(current.get("memory_nodes", 0) or 0),
    }


def compute_relation_stats_metrics(relation_samples: list[dict[str, Any]]) -> dict[str, float]:
    if not relation_samples:
        return {
            "relation_edges_total_now": 0.0,
            "relation_candidates_accepted_now": 0.0,
            "relation_candidates_rejected_now": 0.0,
        }
    valid = [row for row in relation_samples if bool(row.get("ok"))]
    row = (valid[-1] if valid else relation_samples[-1]) or {}
    counts = row.get("candidate_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    return {
        "relation_edges_total_now": float(row.get("memory_edges_total", 0) or 0),
        "relation_candidates_accepted_now": float(counts.get("accepted", 0) or 0),
        "relation_candidates_rejected_now": float(counts.get("rejected", 0) or 0),
    }


def compute_retrieve_baseline_metrics(retrieve_samples: list[dict[str, Any]]) -> dict[str, float]:
    if not retrieve_samples:
        return {"retrieve_p95_baseline_ms": 0.0}
    ok_samples = [s for s in retrieve_samples if bool(s.get("ok"))]
    if not ok_samples:
        return {"retrieve_p95_baseline_ms": 0.0}
    window = max(5, int(len(ok_samples) * 0.2))
    baseline_slice = ok_samples[:window]
    latencies = [float(s.get("latency_ms", 0.0) or 0.0) for s in baseline_slice]
    return {"retrieve_p95_baseline_ms": round(percentile(latencies, 95), 3)}


def compute_relation_precision_metric(run_dir: Path) -> dict[str, float]:
    path = run_dir / "relation_precision_manual.json"
    if not path.exists():
        return {
            "relation_precision_sample": 1.0,
            "relation_precision_sample_size": 0.0,
        }
    data = _read_json(path)
    precision = float(data.get("precision", 0.0) or 0.0)
    sample_size = float(data.get("sample_size", 0) or 0)
    precision = max(0.0, min(1.0, precision))
    return {
        "relation_precision_sample": round(precision, 6),
        "relation_precision_sample_size": sample_size,
    }


def evaluate_gate(metrics: dict[str, float], gate: str = "strict") -> dict[str, Any]:
    if gate != "strict":
        raise ValueError(f"Unsupported gate: {gate}")

    retrieve_p95_baseline = float(metrics.get("retrieve_p95_baseline_ms", 0.0))
    retrieve_p95_dynamic_max = (
        retrieve_p95_baseline * 1.2 if retrieve_p95_baseline > 0 else STRICT_THRESHOLDS["retrieve_p95_ms_max"]
    )

    checks = [
        ("queue_error_rate", "<=", STRICT_THRESHOLDS["queue_error_rate_max"]),
        ("queue_backlog_stall_minutes", "<=", STRICT_THRESHOLDS["queue_backlog_stall_minutes_max"]),
        ("retrieve_p95_ms", "<=", retrieve_p95_dynamic_max),
        ("retrieve_p99_ms", "<=", STRICT_THRESHOLDS["retrieve_p99_ms_max"]),
        ("retrieve_error_rate", "<=", STRICT_THRESHOLDS["retrieve_error_rate_max"]),
        ("promotion_rate", ">=", STRICT_THRESHOLDS["promotion_rate_min"]),
        ("noise_rate", "<=", STRICT_THRESHOLDS["noise_rate_max"]),
        ("session_leak_count", "<=", STRICT_THRESHOLDS["session_leak_count_max"]),
        ("memory_singleton_reduction", ">=", STRICT_THRESHOLDS["memory_singleton_reduction_min"]),
        ("relation_precision_sample", ">=", STRICT_THRESHOLDS["relation_precision_sample_min"]),
    ]

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for name, op, threshold in checks:
        value = float(metrics.get(name, 0.0))
        passed = (value <= threshold) if op == "<=" else (value >= threshold)
        results.append(
            {
                "metric": name,
                "value": value,
                "operator": op,
                "threshold": threshold,
                "pass": passed,
            }
        )
        if not passed:
            failures.append(f"{name} {value} {op} {threshold} (failed)")

    return {
        "gate": gate,
        "overall_pass": len(failures) == 0,
        "checks": results,
        "failures": failures,
    }


def build_summary(run_dir: Path, gate: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(run_dir / "config.json")
    health_samples = _read_jsonl(run_dir / "health_samples.jsonl")
    retrieve_samples = _read_jsonl(run_dir / "retrieve_samples.jsonl")
    worker_samples = _read_jsonl(run_dir / "worker_live_samples.jsonl")
    action_samples = _read_jsonl(run_dir / "memory_action_samples.jsonl")
    synthetic_trace = _read_jsonl(run_dir / "synthetic_trace.jsonl")
    graph_samples = _read_jsonl(run_dir / "graph_samples.jsonl")
    relation_stats_samples = _read_jsonl(run_dir / "relation_stats_samples.jsonl")

    metrics: dict[str, float] = {}
    metrics.update(compute_queue_metrics(health_samples))
    metrics.update(compute_retrieve_metrics(retrieve_samples))
    metrics.update(compute_live_metrics(worker_samples))
    metrics.update(compute_synthetic_metrics(synthetic_trace))
    metrics.update(compute_action_metrics(action_samples))
    metrics.update(compute_graph_metrics(graph_samples))
    metrics.update(compute_relation_stats_metrics(relation_stats_samples))
    metrics.update(compute_retrieve_baseline_metrics(retrieve_samples))
    metrics.update(compute_relation_precision_metric(run_dir))

    passfail = evaluate_gate(metrics, gate=gate)
    summary = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "run_dir": str(run_dir),
        "config": config,
        "sample_counts": {
            "health_samples": len(health_samples),
            "retrieve_samples": len(retrieve_samples),
            "worker_live_samples": len(worker_samples),
            "memory_action_samples": len(action_samples),
            "synthetic_cycles": len(synthetic_trace),
            "graph_samples": len(graph_samples),
            "relation_stats_samples": len(relation_stats_samples),
        },
        "metrics": metrics,
        "gate": passfail,
    }
    return summary, passfail


def _write_summary_markdown(run_dir: Path, summary: dict[str, Any], passfail: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    checks = passfail.get("checks", [])
    lines = [
        "# Burn-in 72h Summary",
        "",
        f"- Generated: `{summary.get('generated_at', '')}`",
        f"- Run dir: `{summary.get('run_dir', '')}`",
        f"- Gate: `{passfail.get('gate', 'strict')}`",
        f"- Overall: `{'PASS' if passfail.get('overall_pass') else 'FAIL'}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in sorted(metrics.keys()):
        lines.append(f"| `{key}` | `{metrics[key]}` |")

    lines.extend(["", "## Gate checks", "", "| Metric | Value | Rule | Status |", "|---|---:|---|---|"])
    for check in checks:
        status = "PASS" if check.get("pass") else "FAIL"
        lines.append(
            f"| `{check.get('metric')}` | `{check.get('value')}` | "
            f"`{check.get('operator')} {check.get('threshold')}` | `{status}` |"
        )

    failures = passfail.get("failures", [])
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"- {item}")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(run_dir: Path, gate: str = "strict") -> tuple[dict[str, Any], dict[str, Any]]:
    summary, passfail = build_summary(run_dir, gate=gate)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "passfail.json").write_text(
        json.dumps(passfail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(run_dir, summary, passfail)
    return summary, passfail


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate burn-in report and pass/fail gate.")
    parser.add_argument("--run-dir", required=True, help="Burn-in run directory")
    parser.add_argument("--gate", default="strict", choices=["strict"], help="Gate profile")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary, passfail = generate_report(run_dir, gate=args.gate)
    print(json.dumps({"run_dir": str(run_dir), "overall_pass": passfail["overall_pass"]}, indent=2))
    return 0 if passfail["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from nas_memory.burnin import report


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


class BurninMetricsTests(unittest.TestCase):
    def test_percentile(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(report.percentile(values, 50), 30.0)
        self.assertGreaterEqual(report.percentile(values, 95), 48.0)
        self.assertGreaterEqual(report.percentile(values, 99), 49.0)

    def test_queue_metrics_and_stall(self) -> None:
        t0 = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc)
        samples = [
            {
                "ts": _iso(t0),
                "queue_counts": {"queued": 0, "processing": 0, "done": 100, "error": 0},
                "oldest_queued_at": None,
            },
            {
                "ts": _iso(t0 + timedelta(minutes=1)),
                "queue_counts": {"queued": 4, "processing": 0, "done": 104, "error": 0},
                "oldest_queued_at": _iso(t0 - timedelta(minutes=10)),
            },
            {
                "ts": _iso(t0 + timedelta(minutes=13)),
                "queue_counts": {"queued": 4, "processing": 0, "done": 108, "error": 1},
                "oldest_queued_at": _iso(t0 - timedelta(minutes=10)),
            },
            {
                "ts": _iso(t0 + timedelta(minutes=14)),
                "queue_counts": {"queued": 0, "processing": 0, "done": 109, "error": 1},
                "oldest_queued_at": None,
            },
        ]
        metrics = report.compute_queue_metrics(samples)
        self.assertAlmostEqual(metrics["queue_error_rate"], 1 / 10, places=3)
        self.assertGreaterEqual(metrics["queue_backlog_stall_minutes"], 10.0)

    def test_live_metrics(self) -> None:
        samples = [
            {"candidates": 2, "promoted": 1, "error": ""},
            {"candidates": 3, "promoted": 1, "error": "claude -p exit 1"},
        ]
        metrics = report.compute_live_metrics(samples)
        self.assertAlmostEqual(metrics["promotion_rate"], 0.4, places=3)
        self.assertAlmostEqual(metrics["noise_rate"], 0.6, places=3)
        self.assertAlmostEqual(metrics["live_extract_error_rate"], 0.5, places=3)

    def test_graph_singleton_reduction_metrics(self) -> None:
        samples = [
            {"ok": True, "memory_nodes": 100, "memory_singletons": 80},
            {"ok": True, "memory_nodes": 100, "memory_singletons": 44},
        ]
        metrics = report.compute_graph_metrics(samples)
        self.assertEqual(metrics["memory_singletons_baseline"], 80.0)
        self.assertEqual(metrics["memory_singletons_now"], 44.0)
        self.assertAlmostEqual(metrics["memory_singleton_reduction"], 0.45, places=3)

    def test_gate_strict_pass_and_fail(self) -> None:
        pass_metrics = {
            "queue_error_rate": 0.001,
            "queue_backlog_stall_minutes": 2.0,
            "retrieve_p95_baseline_ms": 300.0,
            "retrieve_p95_ms": 340.0,
            "retrieve_p99_ms": 600.0,
            "retrieve_error_rate": 0.0,
            "promotion_rate": 0.4,
            "noise_rate": 0.6,
            "session_leak_count": 0.0,
            "memory_singleton_reduction": 0.45,
            "relation_precision_sample": 0.9,
        }
        self.assertTrue(report.evaluate_gate(pass_metrics, gate="strict")["overall_pass"])

        failure_cases = {
            "queue_error_rate": 0.02,
            "queue_backlog_stall_minutes": 30.0,
            "retrieve_p95_ms": 900.0,
            "retrieve_p99_ms": 1800.0,
            "retrieve_error_rate": 0.10,
            "promotion_rate": 0.01,
            "noise_rate": 0.99,
            "session_leak_count": 2.0,
            "memory_singleton_reduction": 0.1,
            "relation_precision_sample": 0.4,
        }
        for metric, bad_value in failure_cases.items():
            with self.subTest(metric=metric):
                failed = dict(pass_metrics)
                failed[metric] = bad_value
                result = report.evaluate_gate(failed, gate="strict")
                self.assertFalse(result["overall_pass"])
                self.assertTrue(any(metric in msg for msg in result["failures"]))


if __name__ == "__main__":
    unittest.main()

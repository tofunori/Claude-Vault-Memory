from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nas_memory.db import get_relation_stats, init_db, upsert_memory_node_versioned
from nas_memory.worker import _handle_relation_compact


class RelationCompactWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory_queue.db"
        init_db(self.db_path)

        upsert_memory_node_versioned(
            self.db_path,
            global_key="g1",
            scope="working_memory",
            fact_text="On utilise FastAPI et SQLite WAL sur NAS.",
            fact_type="fact",
            confidence=0.8,
            source="session_stop",
            relation_mode="same",
            evidence_increment=1,
        )
        upsert_memory_node_versioned(
            self.db_path,
            global_key="g2",
            scope="working_memory",
            fact_text="La stack mémoire NAS retenue est FastAPI avec SQLite WAL.",
            fact_type="fact",
            confidence=0.82,
            source="session_stop",
            relation_mode="same",
            evidence_increment=1,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _settings(self, relation_write: bool) -> SimpleNamespace:
        return SimpleNamespace(
            queue_db_path=self.db_path,
            relation_enable=True,
            relation_write=relation_write,
            relation_batch_max_pairs=200,
            relation_min_confidence=0.6,
            relation_max_new_edges_per_run=50,
            relation_llm_timeout=0,
        )

    def test_shadow_mode_only_candidates(self) -> None:
        _handle_relation_compact(self._settings(False), payload={"actor": "test"}, event_id="evt-shadow")
        stats = get_relation_stats(self.db_path)
        self.assertGreaterEqual(stats["candidate_counts"].get("accepted", 0), 1)
        self.assertEqual(stats["memory_edges_total"], 0)

    def test_write_mode_creates_edges_and_audit(self) -> None:
        _handle_relation_compact(self._settings(True), payload={"actor": "test"}, event_id="evt-write")
        stats = get_relation_stats(self.db_path)
        self.assertGreaterEqual(stats["candidate_counts"].get("accepted", 0), 1)
        self.assertGreaterEqual(stats["memory_edges_total"], 1)

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM memory_audit WHERE action='relation_link'"
            ).fetchone()
            self.assertGreaterEqual(int(row[0]), 1)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

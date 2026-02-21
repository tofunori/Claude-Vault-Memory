from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nas_memory.db import init_db, upsert_memory_node_versioned


class ProfileVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory_queue.db"
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_same_updates_evidence_without_new_version(self) -> None:
        first = upsert_memory_node_versioned(
            self.db_path,
            global_key="g1",
            scope="global_profile",
            fact_text="On garde FastAPI sur NAS",
            fact_type="decision",
            confidence=0.7,
            source="session_stop",
            relation_mode="same",
            evidence_increment=1,
        )
        second = upsert_memory_node_versioned(
            self.db_path,
            global_key="g1",
            scope="global_profile",
            fact_text="On garde FastAPI sur NAS",
            fact_type="decision",
            confidence=0.8,
            source="session_stop",
            relation_mode="same",
            evidence_increment=2,
        )
        self.assertEqual(first["version"], 1)
        self.assertEqual(second["version"], 1)
        self.assertEqual(second["action"], "updated")

    def test_updates_creates_new_version(self) -> None:
        upsert_memory_node_versioned(
            self.db_path,
            global_key="g2",
            scope="global_profile",
            fact_text="On utilise SQLite WAL",
            fact_type="fact",
            confidence=0.6,
            source="session_stop",
            relation_mode="same",
            evidence_increment=1,
        )
        second = upsert_memory_node_versioned(
            self.db_path,
            global_key="g2",
            scope="global_profile",
            fact_text="On utilise SQLite WAL + compaction",
            fact_type="fact",
            confidence=0.7,
            source="session_stop",
            relation_mode="updates",
            evidence_increment=1,
        )
        self.assertEqual(second["action"], "versioned")
        self.assertEqual(second["from_version"], 1)
        self.assertEqual(second["to_version"], 2)
        self.assertEqual(second["edge_relation"], "updates")

    def test_contradicts_creates_new_version(self) -> None:
        upsert_memory_node_versioned(
            self.db_path,
            global_key="g3",
            scope="working_memory",
            fact_text="On active la compaction",
            fact_type="decision",
            confidence=0.7,
            source="turn_live",
            relation_mode="same",
            evidence_increment=1,
        )
        second = upsert_memory_node_versioned(
            self.db_path,
            global_key="g3",
            scope="working_memory",
            fact_text="On n'active pas la compaction",
            fact_type="decision",
            confidence=0.8,
            source="turn_live",
            relation_mode="contradicts",
            evidence_increment=1,
        )
        self.assertEqual(second["action"], "versioned")
        self.assertEqual(second["edge_relation"], "contradicts")
        self.assertEqual(second["to_version"], 2)


if __name__ == "__main__":
    unittest.main()


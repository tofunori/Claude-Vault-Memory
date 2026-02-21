from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nas_memory.db import (
    forget_memory_node,
    get_memory_node_by_id,
    init_db,
    restore_memory_node,
    upsert_memory_node_versioned,
)


class ForgetRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory_queue.db"
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_forget_then_restore(self) -> None:
        created = upsert_memory_node_versioned(
            self.db_path,
            global_key="g-forget",
            scope="global_profile",
            fact_text="Préférence stable: NAS memory",
            fact_type="preference",
            confidence=0.75,
            source="admin",
            relation_mode="same",
            evidence_increment=1,
        )
        node_id = str(created["node_id"])
        f = forget_memory_node(self.db_path, memory_id=node_id, reason="obsolete")
        self.assertEqual(f["action"], "forgotten")
        row = get_memory_node_by_id(self.db_path, node_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "forgotten")

        r = restore_memory_node(self.db_path, memory_id=node_id)
        self.assertEqual(r["action"], "restored")
        row2 = get_memory_node_by_id(self.db_path, node_id)
        self.assertIsNotNone(row2)
        self.assertEqual(row2["status"], "active")


if __name__ == "__main__":
    unittest.main()


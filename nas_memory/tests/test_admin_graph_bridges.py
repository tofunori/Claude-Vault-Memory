from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from nas_memory.db import init_db, upsert_memory_node_versioned
from nas_memory.graph_view import build_unified_graph


@dataclass
class _Settings:
    queue_db_path: Path
    graph_cache_path: Path


class AdminGraphBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.db_path = self.base / "memory_queue.db"
        self.graph_cache_path = self.base / "vault_graph_cache.json"
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fact_text_wikilink_creates_bridge_edge(self) -> None:
        created = upsert_memory_node_versioned(
            self.db_path,
            global_key="bridge-case",
            scope="global_profile",
            fact_text="Décision finale: conserver [[note-x]] comme référence centrale.",
            fact_type="decision",
            confidence=0.9,
            source="admin",
            relation_mode="same",
            evidence_increment=1,
        )
        self.graph_cache_path.write_text(
            json.dumps(
                {
                    "outbound": {
                        "note-x": [],
                    },
                    "backlinks": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        payload = build_unified_graph(_Settings(self.db_path, self.graph_cache_path))
        self.assertEqual(payload["status"], "ok")

        bridges = [e for e in payload["edges"] if e["relation"] == "references_note"]
        self.assertGreaterEqual(len(bridges), 1)

        expected_source = f"mem:{created['node_id']}"
        self.assertTrue(
            any(edge["source"] == expected_source and edge["target"] == "note:note-x" for edge in bridges)
        )


if __name__ == "__main__":
    unittest.main()

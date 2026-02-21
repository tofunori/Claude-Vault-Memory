from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from nas_memory.db import init_db, insert_memory_edge, upsert_memory_node_versioned
from nas_memory.graph_view import build_unified_graph


@dataclass
class _Settings:
    queue_db_path: Path
    graph_cache_path: Path


class AdminGraphPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.db_path = self.base / "memory_queue.db"
        self.graph_cache_path = self.base / "vault_graph_cache.json"
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_payload_contains_nodes_edges_and_stats(self) -> None:
        first = upsert_memory_node_versioned(
            self.db_path,
            global_key="g-a",
            scope="global_profile",
            fact_text="Decision: mémoire centralisée",
            fact_type="decision",
            confidence=0.8,
            source="admin",
            relation_mode="same",
            evidence_increment=1,
        )
        second = upsert_memory_node_versioned(
            self.db_path,
            global_key="g-b",
            scope="working_memory",
            fact_text="Contrainte: éviter double-écriture",
            fact_type="constraint",
            confidence=0.7,
            source="session_stop",
            relation_mode="same",
            evidence_increment=1,
        )
        insert_memory_edge(
            self.db_path,
            src_node_id=first["node_id"],
            dst_node_id=second["node_id"],
            relation="supports",
            confidence=0.72,
        )
        self.graph_cache_path.write_text(
            json.dumps(
                {
                    "outbound": {
                        "note-a": ["note-b"],
                        "note-b": [],
                    },
                    "backlinks": {"note-b": ["note-a"]},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        payload = build_unified_graph(_Settings(self.db_path, self.graph_cache_path))
        self.assertEqual(payload["status"], "ok")
        self.assertIn("generated_at", payload)
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)
        self.assertIn("stats", payload)
        self.assertEqual(payload["stats"]["note_nodes"], 2)
        self.assertEqual(payload["stats"]["note_edges"], 1)
        self.assertGreaterEqual(payload["stats"]["memory_nodes"], 2)
        self.assertGreaterEqual(payload["stats"]["memory_edges"], 1)

        relations = {edge["relation"] for edge in payload["edges"]}
        self.assertIn("links_to", relations)
        self.assertIn("supports", relations)


if __name__ == "__main__":
    unittest.main()

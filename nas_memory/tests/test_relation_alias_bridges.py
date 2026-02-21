from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from nas_memory.db import init_db, upsert_memory_node_versioned, upsert_note_alias
from nas_memory.graph_view import build_unified_graph


@dataclass
class _Settings:
    queue_db_path: Path
    graph_cache_path: Path


class RelationAliasBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.db_path = self.base / "memory_queue.db"
        self.graph_path = self.base / "vault_graph_cache.json"
        init_db(self.db_path)
        self.graph_path.write_text(
            json.dumps({"outbound": {"note-alpha": []}}, ensure_ascii=False),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_alias_creates_reference_bridge(self) -> None:
        upsert_memory_node_versioned(
            self.db_path,
            global_key="g-alias",
            scope="global_profile",
            fact_text="Priorité: appliquer la contrainte alpha project pour cette mémoire.",
            fact_type="constraint",
            confidence=0.75,
            source="admin",
            relation_mode="same",
            evidence_increment=1,
        )
        upsert_note_alias(self.db_path, alias="alpha project", note_id="note-alpha", source="test")

        payload = build_unified_graph(_Settings(self.db_path, self.graph_path))
        bridges = [
            e
            for e in payload["edges"]
            if e.get("relation") == "references_note"
            and e.get("target") == "note:note-alpha"
        ]
        self.assertTrue(bridges)


if __name__ == "__main__":
    unittest.main()

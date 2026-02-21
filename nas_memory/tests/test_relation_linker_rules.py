from __future__ import annotations

import unittest
from types import SimpleNamespace

from nas_memory.relation_linker import generate_relation_candidates


class RelationLinkerRulesTests(unittest.TestCase):
    def _settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            relation_batch_max_pairs=200,
            relation_min_confidence=0.6,
            relation_llm_timeout=0,
        )

    def test_detects_supports(self) -> None:
        nodes = [
            {"id": "a", "fact_text": "On utilise FastAPI et SQLite WAL sur NAS pour la mémoire."},
            {"id": "b", "fact_text": "La stack mémoire NAS retenue est FastAPI avec SQLite WAL."},
        ]
        out = generate_relation_candidates(nodes, self._settings())
        relations = {x["relation"] for x in out["accepted"]}
        self.assertIn("supports", relations)

    def test_detects_contradicts(self) -> None:
        nodes = [
            {"id": "a", "fact_text": "On active la compaction mémoire en production."},
            {"id": "b", "fact_text": "On n'active pas la compaction mémoire en production."},
        ]
        out = generate_relation_candidates(nodes, self._settings())
        relations = {x["relation"] for x in out["accepted"]}
        self.assertIn("contradicts", relations)

    def test_detects_updates_direction(self) -> None:
        nodes = [
            {"id": "a", "fact_text": "Stack mémoire: FastAPI SQLite WAL."},
            {
                "id": "b",
                "fact_text": "Stack mémoire NAS: FastAPI SQLite WAL avec compacteur relationnel batch horaire.",
            },
        ]
        out = generate_relation_candidates(nodes, self._settings())
        updates = [x for x in out["accepted"] if x["relation"] == "updates"]
        self.assertTrue(updates)
        # more specific sentence should point to the more generic one
        self.assertEqual(updates[0]["src_node_id"], "b")
        self.assertEqual(updates[0]["dst_node_id"], "a")


if __name__ == "__main__":
    unittest.main()

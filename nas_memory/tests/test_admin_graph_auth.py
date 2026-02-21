from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore


class AdminGraphAuthTests(unittest.TestCase):
    @unittest.skipIf(TestClient is None, "fastapi dependencies not installed in this interpreter")
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.prev_env = {k: os.environ.get(k) for k in [
            "MEMORY_API_TOKEN",
            "MEMORY_CORE_CONFIG",
            "MEMORY_ROOT",
            "MEMORY_STATE_DIR",
            "MEMORY_QUEUE_DB",
        ]}

        os.environ["MEMORY_API_TOKEN"] = "test-token"
        os.environ["MEMORY_CORE_CONFIG"] = str(self.base / "missing-config.py")
        os.environ["MEMORY_ROOT"] = str(self.base / "memory")
        os.environ["MEMORY_STATE_DIR"] = str(self.base / "memory" / "state")
        os.environ["MEMORY_QUEUE_DB"] = str(self.base / "memory" / "state" / "memory_queue.db")

        memory_root = Path(os.environ["MEMORY_ROOT"])
        memory_root.mkdir(parents=True, exist_ok=True)
        (memory_root / "vault_graph_cache.json").write_text(
            json.dumps({"outbound": {}, "backlinks": {}}, ensure_ascii=False),
            encoding="utf-8",
        )

        for mod in ["nas_memory.api", "nas_memory.security", "nas_memory.config"]:
            if mod in sys.modules:
                del sys.modules[mod]
        self.api_module = importlib.import_module("nas_memory.api")
        self.client = TestClient(self.api_module.app)

    def tearDown(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_admin_graph_requires_auth(self) -> None:
        response = self.client.get("/admin/graph")
        self.assertEqual(response.status_code, 401)

    def test_admin_graph_works_with_token(self) -> None:
        response = self.client.get("/admin/graph", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("nodes", payload)
        self.assertIn("edges", payload)
        self.assertIn("stats", payload)

    def test_admin_graph_ui_requires_auth(self) -> None:
        response = self.client.get("/admin/graph/ui")
        self.assertEqual(response.status_code, 401)

    def test_admin_graph_ui_works_with_query_token(self) -> None:
        response = self.client.get("/admin/graph/ui?token=test-token")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Unified Memory Graph", response.text)


if __name__ == "__main__":
    unittest.main()

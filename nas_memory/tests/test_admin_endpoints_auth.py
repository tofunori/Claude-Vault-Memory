from __future__ import annotations

import os
import unittest

try:
    from fastapi import HTTPException
    from starlette.requests import Request
    from nas_memory import security
except Exception:  # pragma: no cover
    HTTPException = Exception  # type: ignore
    Request = object  # type: ignore
    security = None


def _request(ip: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/health",
        "raw_path": b"/health",
        "query_string": b"",
        "headers": [],
        "client": (ip, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


class AdminAuthTests(unittest.TestCase):
    @unittest.skipIf(security is None, "fastapi dependencies not installed in this interpreter")
    def setUp(self) -> None:
        self.prev_token = os.environ.get("MEMORY_API_TOKEN")
        os.environ["MEMORY_API_TOKEN"] = "test-token"
        security._settings.cache_clear()

    def tearDown(self) -> None:
        if self.prev_token is None:
            os.environ.pop("MEMORY_API_TOKEN", None)
        else:
            os.environ["MEMORY_API_TOKEN"] = self.prev_token
        security._settings.cache_clear()

    def test_rejects_missing_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            security.verify_request_security(_request(), authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_accepts_valid_token(self) -> None:
        security.verify_request_security(_request(), authorization="Bearer test-token")


if __name__ == "__main__":
    unittest.main()

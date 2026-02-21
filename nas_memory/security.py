from __future__ import annotations

import hmac
from functools import lru_cache

from fastapi import Header, HTTPException, Request, status

from .config import Settings, load_settings


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_settings()


def _parse_bearer(auth_header: str | None) -> str | None:
    if not auth_header:
        return None
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header[7:].strip()


def verify_request_security(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    settings = _settings()

    if not settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MEMORY_API_TOKEN is not configured",
        )

    provided = _parse_bearer(authorization)
    if not provided or not hmac.compare_digest(provided, settings.api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token",
        )

    if settings.allowed_ips:
        client_ip = request.client.host if request.client else None
        allowed = set(settings.allowed_ips)
        allowed.update({"127.0.0.1", "::1"})
        if client_ip not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Client IP not allowed: {client_ip}",
            )

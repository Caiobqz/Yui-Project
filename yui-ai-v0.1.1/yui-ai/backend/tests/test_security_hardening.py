"""Testes das correções de segurança da auditoria v0.3.x."""
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from tests.conftest import register_and_login


async def test_expired_token_is_rejected(client: AsyncClient) -> None:
    settings = get_settings()
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": datetime.now(UTC) - timedelta(hours=2),
            "exp": datetime.now(UTC) - timedelta(hours=1),
        },
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    resp = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}
    )
    assert resp.status_code == 401


async def test_token_signed_with_wrong_key_is_rejected(client: AsyncClient) -> None:
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(hours=1),
        },
        "chave-errada-de-um-atacante-1234567890",
        algorithm="HS256",
    )
    resp = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"}
    )
    assert resp.status_code == 401


async def test_login_is_rate_limited_per_ip(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_auth_per_minute", 3)

    for _ in range(3):
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "x@example.com", "password": "senha-qualquer"},
        )
        assert resp.status_code == 401  # credenciais inválidas, mas permitido
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "x@example.com", "password": "senha-qualquer"},
    )
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "60"


async def test_login_response_identical_for_unknown_email_and_wrong_password(
    client: AsyncClient,
) -> None:
    await register_and_login(client)
    wrong_password = await client.post(
        "/api/v1/auth/login",
        json={"email": "leo@example.com", "password": "senha-errada"},
    )
    unknown_email = await client.post(
        "/api/v1/auth/login",
        json={"email": "ninguem@example.com", "password": "senha-errada"},
    )
    # Mesmo status e mesmo corpo (o custo de bcrypt também é equalizado
    # com um hash dummy — não assertável por tempo em teste unitário).
    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


async def test_security_headers_present(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"

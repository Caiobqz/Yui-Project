"""Fixtures de teste: banco SQLite em memória, app com dependências dublês.

Os testes de API exercitam as rotas reais (auth, memórias, chat, tarefas)
contra um banco efêmero, sem Postgres/Redis externos. Extração de memórias e
sumarização pós-turno ficam DESLIGADAS por env (determinismo); os testes
desses componentes os exercitam diretamente.
"""
import os

os.environ.setdefault("MEMORY_EXTRACTION_ENABLED", "false")
os.environ.setdefault("SUMMARIZATION_ENABLED", "false")

from collections.abc import AsyncGenerator  # noqa: E402

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.agents.memory_agent import MemoryAgent  # noqa: E402
from app.agents.yui_core import YuiCore  # noqa: E402
from app.api.deps import get_memory_agent, get_yui_core  # noqa: E402
from app.api.routes.auth import enforce_auth_rate_limit  # noqa: E402
from app.database.session import get_db_session  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.memory.short_term import ShortTermMemory  # noqa: E402
from app.models import Base  # noqa: E402
from app.services.rate_limiter import RateLimiter  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402
from tests.fakes import FakeLLM, FakeRedis  # noqa: E402


@pytest.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    return async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(session_factory) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def client(session_factory, fake_redis, fake_llm) -> AsyncGenerator[AsyncClient, None]:
    async def override_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def override_yui_core() -> YuiCore:
        return YuiCore(
            session_factory=session_factory,
            short_term=ShortTermMemory(fake_redis),  # type: ignore[arg-type]
            llm=fake_llm,
            rate_limiter=RateLimiter(fake_redis),  # type: ignore[arg-type]
            embeddings=None,
            registry=build_default_registry(),
        )

    def override_memory_agent() -> MemoryAgent:
        return MemoryAgent(
            llm=fake_llm, embeddings=None, session_factory=session_factory
        )

    async def override_auth_rate_limit() -> None:
        # O rate limit de auth por IP usa o Redis real via get_redis();
        # nos testes usa o FakeRedis compartilhado (limites default folgados).
        limiter = RateLimiter(fake_redis)  # type: ignore[arg-type]
        from app.core.config import get_settings

        await limiter.enforce_fixed_window(
            key="auth:test-client",
            limit=get_settings().rate_limit_auth_per_minute,
            window_seconds=60,
            message="Muitas tentativas de autenticação. Aguarde um minuto.",
            retry_after_seconds=60,
        )

    fastapi_app.dependency_overrides[get_db_session] = override_db_session
    fastapi_app.dependency_overrides[get_yui_core] = override_yui_core
    fastapi_app.dependency_overrides[get_memory_agent] = override_memory_agent
    fastapi_app.dependency_overrides[enforce_auth_rate_limit] = override_auth_rate_limit
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    fastapi_app.dependency_overrides.clear()


async def register_and_login(
    client: AsyncClient, email: str = "leo@example.com", password: str = "senha-segura-1"
) -> dict[str, str]:
    """Cria um usuário e retorna o header Authorization pronto."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "name": "Leo"},
    )
    assert resp.status_code == 201, resp.text
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

"""Fixtures de teste: banco SQLite em memória, app com dependências dublês.

Os testes de API exercitam as rotas reais (auth, memórias, chat) contra um
banco efêmero, sem Postgres/Redis externos — possível porque os modelos usam
o tipo portável sqlalchemy.Uuid e o YuiCore recebe dependências injetadas.
"""
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.agents.yui_core import YuiCore
from app.api.deps import get_yui_core
from app.database.session import get_db_session
from app.main import app as fastapi_app
from app.memory.short_term import ShortTermMemory
from app.models import Base
from app.services.rate_limiter import RateLimiter
from tests.fakes import FakeLLM, FakeRedis


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
        )

    fastapi_app.dependency_overrides[get_db_session] = override_db_session
    fastapi_app.dependency_overrides[get_yui_core] = override_yui_core
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

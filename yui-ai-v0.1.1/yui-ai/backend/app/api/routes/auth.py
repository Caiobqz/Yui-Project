"""Registro, login e perfil do usuário autenticado.

Proteções desta rota:
- Rate limit por IP em register/login (anti brute force / spam de contas).
  Atrás de proxy reverso, rode o uvicorn com --proxy-headers para que
  request.client reflita o IP real.
- Resposta 401 idêntica (mensagem E tempo) para e-mail inexistente e senha
  errada: um bcrypt "dummy" é verificado quando o usuário não existe, para
  não permitir enumeração de e-mails por diferença de latência.
"""
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import (
    LoginRequest,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
)
from app.core.config import get_settings
from app.core.security import create_access_token, hash_password, verify_password
from app.database.redis_client import get_redis
from app.models.user import User
from app.services.rate_limiter import RateLimiter


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """Hash usado para igualar o tempo do login quando o e-mail não existe."""
    return hash_password("yui-timing-equalizer")


async def enforce_auth_rate_limit(request: Request) -> None:
    """Janela fixa por IP para os endpoints de autenticação."""
    client_ip = request.client.host if request.client else "unknown"
    limiter = RateLimiter(await get_redis())
    await limiter.enforce_fixed_window(
        key=f"auth:{client_ip}",
        limit=get_settings().rate_limit_auth_per_minute,
        window_seconds=60,
        message="Muitas tentativas de autenticação. Aguarde um minuto.",
        retry_after_seconds=60,
    )


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    dependencies=[Depends(enforce_auth_rate_limit)],
)

_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Credenciais inválidas.",
)


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegisterRequest, session: DbSession) -> UserResponse:
    email = payload.email.lower()
    existing = await session.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="E-mail já cadastrado.",
        )

    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        name=payload.name,
    )
    session.add(user)
    await session.flush()
    return UserResponse.model_validate(user)


@router.post("/login")
async def login(payload: LoginRequest, session: DbSession) -> TokenResponse:
    user = await session.scalar(
        select(User).where(User.email == payload.email.lower())
    )
    if user is None:
        # Custo de bcrypt equivalente ao caminho de senha errada.
        verify_password(payload.password, _dummy_password_hash())
        raise _INVALID_CREDENTIALS
    if not user.is_active or not verify_password(
        payload.password, user.hashed_password
    ):
        raise _INVALID_CREDENTIALS
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me")
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)

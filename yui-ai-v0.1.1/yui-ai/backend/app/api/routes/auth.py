"""Registro, login e perfil do usuário autenticado."""
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import (
    LoginRequest,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


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
    # Mensagem única para e-mail inexistente e senha errada: não revela
    # quais e-mails estão cadastrados.
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.hashed_password)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas.",
        )
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me")
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)

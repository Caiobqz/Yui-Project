"""Dependências injetáveis da API."""
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.yui_core import YuiCore
from app.database.redis_client import get_redis
from app.database.session import get_db_session
from app.memory.short_term import ShortTermMemory
from app.services.llm.factory import get_llm_provider

DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_yui_core(session: DbSession) -> YuiCore:
    redis = await get_redis()
    return YuiCore(
        session=session,
        short_term=ShortTermMemory(redis),
        llm=get_llm_provider(),
    )


Yui = Annotated[YuiCore, Depends(get_yui_core)]

"""Configurações centrais da aplicação, carregadas de variáveis de ambiente."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Yui AI Assistant"
    version: str = "0.1.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    # CORS — origens do frontend, separadas por vírgula no .env
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Infraestrutura
    database_url: str = "postgresql+asyncpg://yui:yui_password@localhost:5432/yui"
    # Loga SQL no console. Atenção: expõe conteúdo das conversas nos logs.
    database_echo: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Provedor de IA
    llm_provider: str = "claude"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7

    # Memória
    short_term_max_messages: int = 20
    short_term_ttl_seconds: int = 86400
    memory_retrieval_limit: int = 5

    # Personalidade
    personality_path: str = "app/config/personality.yaml"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Instância única de Settings (cacheada por processo)."""
    return Settings()

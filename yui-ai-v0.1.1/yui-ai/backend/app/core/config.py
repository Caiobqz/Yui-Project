"""Configurações centrais da aplicação, carregadas de variáveis de ambiente."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do backend (.../backend), usada para resolver caminhos relativos
# independentemente do diretório de onde o processo foi iniciado.
BASE_DIR = Path(__file__).resolve().parents[2]

# Valor default apenas para desenvolvimento local. validate_runtime() impede
# que a aplicação suba em produção com este valor.
_DEV_JWT_SECRET = "dev-secret-inseguro-nao-use-em-producao"


class ConfigurationError(RuntimeError):
    """Configuração essencial ausente ou insegura, detectada no boot."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Yui AI Assistant"
    version: str = "0.3.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    # CORS — origens do frontend, separadas por vírgula no .env
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # Infraestrutura
    database_url: str = "postgresql+asyncpg://yui:yui_password@localhost:5432/yui"
    # Loga SQL no console. Atenção: expõe conteúdo das conversas nos logs.
    database_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout_seconds: int = 10
    redis_url: str = "redis://localhost:6379/0"

    # Autenticação
    jwt_secret_key: str = _DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Provedor de IA
    llm_provider: str = "claude"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    # Modelo utilitário: trabalho cognitivo de bastidor (extração de
    # memórias, adaptação, resumo) roda num modelo mais barato.
    anthropic_utility_model: str = "claude-haiku-4-5"
    openai_utility_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Orçamento do histórico enviado ao modelo, em caracteres
    # (aproximação: ~4 caracteres por token).
    llm_max_history_chars: int = 24_000
    # Máximo de iterações do loop de ferramentas por turno.
    llm_max_tool_iterations: int = 5

    # Embeddings / memória semântica (RAG)
    # "disabled" mantém a busca lexical — a Yui funciona sem chave da OpenAI.
    embedding_provider: str = "disabled"  # openai | disabled
    embedding_model: str = "text-embedding-3-small"
    # Similaridade de cosseno mínima para considerar uma memória relevante.
    memory_similarity_threshold: float = 0.35
    # Acima deste valor, uma memória nova é considerada duplicata.
    memory_duplicate_threshold: float = 0.90

    # Extração automática de memórias (pós-turno, em background)
    memory_extraction_enabled: bool = True
    memory_min_confidence: float = 0.6

    # Resumo automático de conversas longas (pós-turno, em background)
    summarization_enabled: bool = True

    # Núcleo cognitivo (v0.3)
    # Curiosity Engine: sugere no máximo 1 pergunta quando detecta lacunas.
    curiosity_enabled: bool = True
    curiosity_min_interactions: int = 3
    # Planos sem progresso há N dias despertam curiosidade.
    plan_stale_days: int = 7
    # Adaptation Engine: teto de notas aprendidas por usuário.
    adaptation_max_notes: int = 12
    # Manutenção de memória: roda a cada N interações do usuário.
    memory_maintenance_interval: int = 20
    # Poda: memórias extraídas, nunca usadas, mais velhas que N dias e com
    # pontuação (importância × recência) abaixo do limiar são removidas.
    memory_prune_min_age_days: int = 30
    memory_prune_score_threshold: float = 0.15

    # Ferramentas
    web_search_enabled: bool = False

    # Memória
    short_term_max_messages: int = 20
    short_term_ttl_seconds: int = 86400
    memory_retrieval_limit: int = 5

    # Rate limiting / custos — valores do plano "free".
    # Planos adicionais entram em app/services/rate_limiter.py.
    rate_limit_chat_per_minute: int = 20
    daily_token_limit: int = 200_000
    # Tentativas de login/registro por IP e minuto (anti brute force).
    rate_limit_auth_per_minute: int = 10

    # Personalidade
    personality_path: str = "app/config/personality.yaml"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def personality_file(self) -> Path:
        path = Path(self.personality_path)
        return path if path.is_absolute() else BASE_DIR / path

    def validate_runtime(self) -> None:
        """Falha cedo se configurações essenciais estiverem ausentes/inseguras.

        Chamado no lifespan da aplicação. Em desenvolvimento os defaults são
        tolerados; fora dele, valores inseguros impedem o boot.
        """
        errors: list[str] = []

        if self.short_term_max_messages % 2 != 0:
            errors.append(
                "SHORT_TERM_MAX_MESSAGES deve ser par: o histórico é gravado em "
                "pares usuário/assistente e um corte ímpar quebra a alternância "
                "exigida pelos provedores de IA."
            )
        if self.llm_provider.lower() not in {"claude", "openai"}:
            errors.append(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. Aceitos: 'claude', 'openai'."
            )
        if self.embedding_provider.lower() not in {"openai", "disabled"}:
            errors.append(
                f"EMBEDDING_PROVIDER inválido: '{self.embedding_provider}'. "
                "Aceitos: 'openai', 'disabled'."
            )
        if not self.is_development:
            if "*" in self.cors_origin_list:
                errors.append(
                    "CORS_ORIGINS não pode conter '*' fora de development "
                    "(a API usa credenciais)."
                )
            if self.jwt_secret_key == _DEV_JWT_SECRET or len(self.jwt_secret_key) < 32:
                errors.append(
                    "JWT_SECRET_KEY ausente ou fraca (mínimo 32 caracteres). "
                    "Gere uma com: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
                )
            if "yui_password" in self.database_url:
                errors.append(
                    "DATABASE_URL usa a senha default de desenvolvimento ('yui_password')."
                )

        if errors:
            raise ConfigurationError(
                "Configuração inválida:\n- " + "\n- ".join(errors)
            )


@lru_cache
def get_settings() -> Settings:
    """Instância única de Settings (cacheada por processo)."""
    return Settings()

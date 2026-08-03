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

# Backends de LLM aceitos por LLM_PROVIDER / UTILITY_LLM_PROVIDER.
_LLM_BACKENDS = {"claude", "openai", "llama_cpp", "ollama"}
# Backends de embeddings aceitos por EMBEDDING_PROVIDER.
_EMBEDDING_BACKENDS = {"openai", "sentence_transformers", "disabled"}


class ConfigurationError(RuntimeError):
    """Configuração essencial ausente ou insegura, detectada no boot."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Yui AI Companion"
    version: str = "0.5.0"
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

    # --- Provedor utilitário: backend e modelo próprios (opcional) ---------
    # Quando ausentes, o utilitário herda o backend/modelo do provider
    # principal (comportamento anterior, preservado). Só relevante para
    # llama_cpp/ollama: Claude/OpenAI já usam anthropic_utility_model /
    # openai_utility_model acima.
    utility_llm_provider: str | None = None
    utility_llm_model_path: str | None = None
    ollama_utility_model: str | None = None

    # --- llama.cpp (inferência local via llama-cpp-python) ------------------
    # Caminho para o arquivo .gguf. Resolvido a partir de BASE_DIR (raiz do
    # backend) quando relativo — nunca depende do diretório de trabalho do
    # processo nem de caminhos específicos da máquina do desenvolvedor.
    llm_model_path: str | None = None
    # Tamanho do contexto (tokens). None deixa a biblioteca decidir a partir
    # dos metadados do modelo.
    llm_context_size: int = 8192
    # Camadas offload para GPU: 0 desliga, -1 tenta offload total, N>0 um
    # número específico de camadas. Depende do runtime (CUDA/Metal/ROCm)
    # com que llama-cpp-python foi compilado — não há fallback silencioso.
    llm_gpu_layers: int = 0
    llm_threads: int = 4
    llm_batch_size: int = 512
    # Reservado para uso futuro caso uma implementação com isolamento de
    # contexto por requisição seja adicionada. NESTA VERSÃO, o
    # LlamaCppProvider SEMPRE trava a concorrência em 1, independentemente
    # deste valor — não há isolamento de contexto por requisição e a
    # biblioteca não garante segurança de uma única instância sob
    # concorrência. Um valor > 1 aqui é ignorado (aviso registrado no log).
    llm_max_concurrent_requests: int = 1
    # Chat format do llama.cpp (ex.: "chatml", "llama-3", "chatml-function-calling").
    # Vazio/None deixa a biblioteca detectar pelos metadados do GGUF.
    # NÃO é usado sozinho como prova de suporte a tool calling: apenas
    # formatos com sufixo "-function-calling" (ou "functionary"/"functionary-v2")
    # são considerados — "chatml" simples NÃO implica suporte a ferramentas.
    # Ver _TOOL_CALLING_CHAT_FORMATS em LlamaCppProvider.
    llm_chat_format: str | None = None
    # Declara explicitamente se o modelo/chat format configurado suporta tool
    # calling. None = detectar via llm_chat_format (lista restrita, ver
    # acima); True/False sobrescreve a detecção.
    llm_supports_tool_calling: bool | None = None

    # --- Ollama (inferência local via servidor HTTP) -------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_seconds: float = 120.0
    # Default CONSERVADOR: quando None, tool_calling é considerado
    # INDISPONÍVEL para qualquer modelo Ollama — não há forma confiável de
    # detectar automaticamente (não exposto de forma estruturada e uniforme
    # por /api/show). Definir True só após validar manualmente que o
    # modelo escolhido suporta tool calling.
    ollama_supports_tool_calling: bool | None = None

    # Embeddings / memória semântica (RAG)
    # "disabled" mantém a busca lexical — a Yui funciona sem provedor externo.
    embedding_provider: str = "disabled"  # openai | sentence_transformers | disabled
    embedding_model: str = "text-embedding-3-small"
    # Dispositivo para sentence-transformers: "cpu" (default seguro) ou
    # "cuda"/"mps" quando o hardware e a instalação do torch suportarem.
    # Sem fallback automático de GPU para CPU: uma configuração de GPU que
    # falhar deve falhar de forma clara, não degradar silenciosamente.
    embedding_device: str = "cpu"
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
    # Após sugerir uma pergunta, silencia por N interações (v0.5): uma lacuna
    # estável nunca deve virar a mesma pergunta em todos os turnos.
    curiosity_min_gap_interactions: int = 4
    # Planos sem progresso há N dias despertam curiosidade.
    plan_stale_days: int = 7
    # Goal Engine: plano parado além disso é considerado abandonado.
    goal_abandoned_days: int = 21

    # Attention Manager (v0.4) — pesos do Attention Score (determinístico).
    attention_goal_weight: float = 0.30
    attention_relationship_weight: float = 0.15
    attention_preference_weight: float = 0.10
    attention_graph_weight: float = 0.15
    attention_redundancy_penalty: float = 0.50

    # Context Orchestrator (v0.4) — orçamento de caracteres por bloco
    # (proxy determinístico de tokens; ~4 chars/token).
    ctx_budget_self_chars: int = 800
    ctx_budget_world_chars: int = 600
    ctx_budget_goals_chars: int = 1200
    ctx_budget_adaptation_chars: int = 1200
    ctx_budget_summary_chars: int = 2400
    ctx_budget_memories_chars: int = 4000
    ctx_budget_affect_chars: int = 400

    # Autonomia (v0.4) — Judgement Engine / Bússola Moral.
    autonomy_enabled: bool = True

    # Companion Core (v0.5) — afeto persistente e iniciativas.
    # Estados afetivos computacionais (warmth/joy/concern): atualizados na
    # fase 3 do turno e injetados no prompt como domínio self.
    affect_enabled: bool = True
    # Geração de iniciativas no pós-turno (a decisão continua no Judgement
    # Engine; este flag governa apenas o registro automático em background).
    initiative_generation_enabled: bool = True
    # Mesma situação (dedupe_key) não é reproposta dentro desta janela.
    initiative_cooldown_days: int = 14
    # Teto de iniciativas pendentes por usuário — iniciativas são RARAS.
    initiative_max_pending: int = 2
    moral_act_threshold: float = 0.45
    moral_confidence_threshold: float = 0.55
    moral_high_risk_threshold: float = 0.7
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

    @property
    def resolved_llm_model_path(self) -> Path | None:
        """Caminho do GGUF, resolvido a partir de BASE_DIR quando relativo.

        Nunca depende do diretório de trabalho do processo — evita o erro
        clássico de "funciona na minha máquina" quando o uvicorn é iniciado
        de um diretório diferente.
        """
        if not self.llm_model_path:
            return None
        path = Path(self.llm_model_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def resolved_utility_llm_model_path(self) -> Path | None:
        if not self.utility_llm_model_path:
            return None
        path = Path(self.utility_llm_model_path)
        return path if path.is_absolute() else BASE_DIR / path

    @property
    def effective_utility_llm_provider(self) -> str:
        """Backend do provider utilitário: o configurado ou, na ausência,
        o mesmo do provider principal (comportamento anterior preservado).
        """
        return (self.utility_llm_provider or self.llm_provider).lower()

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
        llm_provider = self.llm_provider.lower()
        if llm_provider not in _LLM_BACKENDS:
            errors.append(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. Aceitos: "
                + ", ".join(sorted(_LLM_BACKENDS)) + "."
            )
        utility_provider = self.effective_utility_llm_provider
        if (
            self.utility_llm_provider is not None
            and utility_provider not in _LLM_BACKENDS
        ):
            errors.append(
                f"UTILITY_LLM_PROVIDER inválido: '{self.utility_llm_provider}'. "
                "Aceitos: " + ", ".join(sorted(_LLM_BACKENDS)) + "."
            )
        if self.embedding_provider.lower() not in _EMBEDDING_BACKENDS:
            errors.append(
                f"EMBEDDING_PROVIDER inválido: '{self.embedding_provider}'. "
                "Aceitos: " + ", ".join(sorted(_EMBEDDING_BACKENDS)) + "."
            )

        # Validação condicional: só exige o que é relevante ao backend
        # efetivamente selecionado para cada papel (principal/utilitário).
        # Deliberadamente NÃO exige configuração de llama.cpp e Ollama ao
        # mesmo tempo — só o backend em uso precisa estar configurado.
        for role, provider in (("LLM_PROVIDER", llm_provider), ("UTILITY_LLM_PROVIDER", utility_provider)):
            if provider == "llama_cpp":
                is_utility = role == "UTILITY_LLM_PROVIDER"
                path = (
                    self.resolved_utility_llm_model_path
                    if is_utility and self.utility_llm_model_path
                    else self.resolved_llm_model_path
                )
                if path is None:
                    var = "UTILITY_LLM_MODEL_PATH ou LLM_MODEL_PATH" if is_utility else "LLM_MODEL_PATH"
                    errors.append(
                        f"{role}=llama_cpp exige {var} apontando para um arquivo .gguf."
                    )
            elif provider == "ollama":
                if not self.ollama_base_url:
                    errors.append(f"{role}=ollama exige OLLAMA_BASE_URL configurada.")
                if role == "UTILITY_LLM_PROVIDER":
                    if not (self.ollama_utility_model or self.ollama_model):
                        errors.append(
                            "UTILITY_LLM_PROVIDER=ollama exige OLLAMA_UTILITY_MODEL "
                            "ou OLLAMA_MODEL configurado."
                        )
                elif not self.ollama_model:
                    errors.append("LLM_PROVIDER=ollama exige OLLAMA_MODEL configurado.")

        if self.embedding_provider.lower() == "sentence_transformers" and not self.embedding_model:
            errors.append(
                "EMBEDDING_PROVIDER=sentence_transformers exige EMBEDDING_MODEL "
                "(ex.: 'BAAI/bge-small-en-v1.5')."
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

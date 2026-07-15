"""Self Model — modelo interno, SOMENTE LEITURA, da própria Yui.

Contém identidade, versão, capacidades, limitações, ferramentas disponíveis,
estado do sistema, módulos carregados e saúde. É um snapshot congelado
(dataclass frozen) construído a partir de fontes autoritativas (identidade em
código, configuração, registro de ferramentas). Nenhum usuário, agente ou LLM
pode modificá-lo: não há setters e o objeto é imutável.
"""
from dataclasses import dataclass
from functools import lru_cache

from app.cognition.identity import YUI_IDENTITY
from app.core.config import get_settings
from app.tools.registry import ToolRegistry

# Capacidades cognitivas ativas — descritivas, derivadas da arquitetura.
_CAPABILITIES: tuple[str, ...] = (
    "conversa com contexto e memória de longo prazo",
    "memória hierárquica (semântica, episódica, procedural, relacionamento)",
    "recuperação por atenção (relevância, recência, objetivos, redundância)",
    "acompanhamento de objetivos e planos",
    "julgamento de ações autônomas pela Bússola Moral",
    "execução de ferramentas autorizadas",
)

# Módulos cognitivos carregados (para o Self Model e observabilidade).
_MODULES: tuple[str, ...] = (
    "identity_system",
    "memory_system",
    "attention_manager",
    "goal_engine",
    "moral_compass",
    "judgement_engine",
    "world_model",
    "context_orchestrator",
    "permission_system",
    "guardian",
)


@dataclass(frozen=True)
class SelfModel:
    name: str
    version: str
    purpose: str
    capabilities: tuple[str, ...]
    limitations: tuple[str, ...]
    tools_available: tuple[str, ...]
    modules_loaded: tuple[str, ...]
    subsystems: dict[str, bool]
    health: str

    def to_prompt(self) -> str:
        """Bloco do domínio 'self' para o WorldModel (contexto ao modelo)."""
        tools = ", ".join(self.tools_available) or "nenhuma"
        return (
            f"Você é a Yui, versão {self.version}. "
            f"Suas capacidades atuais: {'; '.join(self.capabilities)}. "
            f"Ferramentas disponíveis neste momento: {tools}. "
            "Você conhece seus próprios limites e nunca finge capacidades que "
            "não possui."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "purpose": self.purpose,
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "tools_available": list(self.tools_available),
            "modules_loaded": list(self.modules_loaded),
            "subsystems": self.subsystems,
            "health": self.health,
        }


def build_self_model(registry: ToolRegistry) -> SelfModel:
    """Constrói o snapshot read-only a partir das fontes autoritativas."""
    settings = get_settings()
    subsystems = {
        "embeddings": settings.embedding_provider.lower() != "disabled",
        "web_search": settings.web_search_enabled,
        "memory_extraction": settings.memory_extraction_enabled,
        "summarization": settings.summarization_enabled,
        "curiosity": settings.curiosity_enabled,
        "autonomy": settings.autonomy_enabled,
    }
    return SelfModel(
        name=YUI_IDENTITY.name,
        version=settings.version,
        purpose=YUI_IDENTITY.purpose,
        capabilities=_CAPABILITIES,
        limitations=YUI_IDENTITY.limitations,
        tools_available=tuple(spec.name for spec in registry.specs()),
        modules_loaded=_MODULES,
        subsystems=subsystems,
        health="operational",
    )


@lru_cache(maxsize=1)
def _cached_registry() -> ToolRegistry:
    from app.tools.registry import build_default_registry

    return build_default_registry()


def get_self_model() -> SelfModel:
    """Self Model do processo (registro default de ferramentas)."""
    return build_self_model(_cached_registry())

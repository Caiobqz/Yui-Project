"""Guardian Agent — segurança, privacidade e validação de ações.

Três responsabilidades:
- validar chamadas de ferramenta ANTES da execução (ferramenta existe,
  argumentos obrigatórios presentes);
- triagem de conteúdo de memórias (nunca persistir segredos/credenciais);
- limitar o tamanho de resultados de ferramenta devolvidos ao modelo.
"""
import re

from app.services.llm.base import ToolCall
from app.tools.registry import ToolRegistry

# Padrões de segredos que nunca devem virar memória persistida.
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(senha|password|passwd)\b\s*[:=]?"),
    re.compile(r"(?i)\bapi[_\- ]?key\b"),
    re.compile(r"(?i)\b(token|secret)\b\s*[:=]"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),          # chaves estilo OpenAI/Anthropic
    re.compile(r"\b\d{13,19}\b"),                        # possíveis números de cartão
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

_MAX_TOOL_RESULT_CHARS = 4000
_MAX_MEMORY_CHARS = 1000


class GuardianAgent:
    def validate_tool_call(
        self,
        registry: ToolRegistry,
        call: ToolCall,
        permission_overrides: dict[str, bool] | None = None,
    ) -> str | None:
        """Retorna a mensagem de erro (devolvida ao modelo) ou None se válida.

        Além de existência e argumentos, aplica o Permission System:
        decisão explícita do usuário > default da ferramenta.
        """
        tool = registry.get(call.name)
        if tool is None:
            return f"Ferramenta desconhecida: '{call.name}'. Ela não está autorizada."
        overrides = permission_overrides or {}
        if not overrides.get(call.name, tool.default_allowed):
            return (
                f"A ferramenta '{call.name}' não está autorizada para este "
                "usuário. Informe que ele pode conceder a permissão nas "
                "configurações de permissões."
            )
        if not isinstance(call.arguments, dict):
            return "Argumentos inválidos: esperado um objeto JSON."
        required = tool.spec.input_schema.get("required", [])
        missing = [field for field in required if field not in call.arguments]
        if missing:
            return (
                f"Argumentos obrigatórios ausentes em '{call.name}': "
                f"{', '.join(missing)}."
            )
        return None

    def screen_memory_content(self, content: str) -> str | None:
        """Retorna o motivo da rejeição ou None se o conteúdo pode ser salvo."""
        stripped = content.strip()
        if not stripped:
            return "conteúdo vazio"
        if len(stripped) > _MAX_MEMORY_CHARS:
            return f"conteúdo excede {_MAX_MEMORY_CHARS} caracteres"
        for pattern in _SECRET_PATTERNS:
            if pattern.search(stripped):
                return "conteúdo aparenta conter credenciais ou dados sensíveis"
        return None

    def clamp_tool_result(self, result: str) -> str:
        if len(result) <= _MAX_TOOL_RESULT_CHARS:
            return result
        return result[:_MAX_TOOL_RESULT_CHARS] + "\n[resultado truncado]"

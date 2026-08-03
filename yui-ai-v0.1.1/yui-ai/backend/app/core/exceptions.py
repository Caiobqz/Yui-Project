"""Exceções de domínio da Yui.

Camadas internas (agente, serviços) levantam estas exceções; a conversão
para respostas HTTP acontece nos handlers registrados em app/main.py.
Detalhes internos são logados, nunca expostos ao cliente.
"""


class YuiError(Exception):
    """Base para erros de domínio."""


class ConversationNotFoundError(YuiError):
    """Conversa inexistente ou pertencente a outro usuário."""


class RateLimitExceededError(YuiError):
    """Limite de uso excedido (requisições por minuto ou tokens por dia)."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

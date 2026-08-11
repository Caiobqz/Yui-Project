"""Testes do GuardianAgent: validação de ferramentas e triagem de memórias."""
from app.agents.guardian import GuardianAgent
from app.services.llm.base import ToolCall, ToolSpec
from app.tools.base import Tool
from app.tools.registry import ToolRegistry


async def _noop(ctx, args) -> str:
    return "ok"


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                spec=ToolSpec(
                    name="create_task",
                    description="t",
                    input_schema={
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                ),
                handler=_noop,
            )
        ]
    )


def test_unknown_tool_is_rejected() -> None:
    guardian = GuardianAgent()
    error = guardian.validate_tool_call(
        _registry(), ToolCall(id="1", name="rm_rf", arguments={})
    )
    assert error is not None and "desconhecida" in error


def test_missing_required_arguments_are_rejected() -> None:
    guardian = GuardianAgent()
    error = guardian.validate_tool_call(
        _registry(), ToolCall(id="1", name="create_task", arguments={})
    )
    assert error is not None and "title" in error


def test_valid_call_passes() -> None:
    guardian = GuardianAgent()
    error = guardian.validate_tool_call(
        _registry(), ToolCall(id="1", name="create_task", arguments={"title": "x"})
    )
    assert error is None


def test_memory_screening_blocks_secrets() -> None:
    guardian = GuardianAgent()
    assert guardian.screen_memory_content("Gosta de café") is None
    assert guardian.screen_memory_content("senha: hunter2") is not None
    assert guardian.screen_memory_content("minha api_key nova") is not None
    assert guardian.screen_memory_content("token = abc123") is not None
    assert guardian.screen_memory_content("sk-abcdefghijklmnop1234") is not None
    assert guardian.screen_memory_content("cartão 4111111111111111") is not None
    assert guardian.screen_memory_content("") is not None


def test_memory_screening_blocks_additional_credential_formats() -> None:
    """Sinônimo em PT e formatos de token comuns (ver hardening)."""
    guardian = GuardianAgent()
    assert guardian.screen_memory_content("aqui está minha chave da api") is not None
    assert guardian.screen_memory_content("essa é minha chave secreta") is not None
    assert (
        guardian.screen_memory_content("ghp_abcdefghijklmnopqrstuvwxyz012345") is not None
    )
    assert guardian.screen_memory_content("AKIAABCDEFGHIJKLMNOP") is not None
    assert (
        guardian.screen_memory_content(
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQrb9F"
        )
        is not None
    )


def test_memory_screening_does_not_false_positive_on_the_word_chave() -> None:
    """'Chave' sozinha (sem qualificador de credencial) não deve bloquear."""
    guardian = GuardianAgent()
    assert guardian.screen_memory_content("perdeu a chave de casa de novo") is None
    assert guardian.screen_memory_content("comprou uma chave de fenda nova") is None


def test_tool_result_clamp() -> None:
    guardian = GuardianAgent()
    small = "resultado"
    assert guardian.clamp_tool_result(small) == small
    clamped = guardian.clamp_tool_result("x" * 10_000)
    assert len(clamped) < 10_000
    assert clamped.endswith("[resultado truncado]")

"""Testes de Settings.validate_runtime() para os backends locais
(llama.cpp, Ollama, sentence-transformers) e para a preservação do
comportamento anterior (Claude/OpenAI).
"""
import pytest

from app.core.config import ConfigurationError, Settings


def _dev_settings(**overrides) -> Settings:
    return Settings(environment="development", jwt_secret_key="x" * 32, **overrides)


# --- llama.cpp -----------------------------------------------------------------


def test_llama_cpp_without_model_path_fails(tmp_path) -> None:
    settings = _dev_settings(llm_provider="llama_cpp", llm_model_path=None)
    with pytest.raises(ConfigurationError, match="LLM_MODEL_PATH"):
        settings.validate_runtime()


def test_llama_cpp_with_model_path_passes(tmp_path) -> None:
    settings = _dev_settings(llm_provider="llama_cpp", llm_model_path=str(tmp_path / "m.gguf"))
    settings.validate_runtime()  # não deve levantar


def test_llama_cpp_relative_path_resolved_from_base_dir() -> None:
    settings = _dev_settings(llm_provider="llama_cpp", llm_model_path="models/qwen.gguf")
    resolved = settings.resolved_llm_model_path
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.parts[-2:] == ("models", "qwen.gguf")


# --- Ollama ----------------------------------------------------------------


def test_ollama_without_base_url_fails() -> None:
    settings = _dev_settings(llm_provider="ollama", ollama_base_url="", ollama_model="qwen3:8b")
    with pytest.raises(ConfigurationError, match="OLLAMA_BASE_URL"):
        settings.validate_runtime()


def test_ollama_without_model_fails() -> None:
    settings = _dev_settings(llm_provider="ollama", ollama_base_url="http://localhost:11434", ollama_model="")
    with pytest.raises(ConfigurationError, match="OLLAMA_MODEL"):
        settings.validate_runtime()


def test_ollama_fully_configured_passes() -> None:
    settings = _dev_settings(
        llm_provider="ollama", ollama_base_url="http://localhost:11434", ollama_model="qwen3:8b"
    )
    settings.validate_runtime()


# --- Utilitário: herança e configuração separada -------------------------------


def test_utility_inherits_main_backend_without_error(tmp_path) -> None:
    settings = _dev_settings(llm_provider="llama_cpp", llm_model_path=str(tmp_path / "m.gguf"))
    settings.validate_runtime()
    assert settings.effective_utility_llm_provider == "llama_cpp"
    assert settings.resolved_utility_llm_model_path is None  # herda o do principal na factory


def test_utility_with_separate_backend_does_not_require_cross_config(tmp_path) -> None:
    """Principal llama.cpp + utilitário ollama: não deve exigir configuração
    cruzada (nem GGUF utilitário, nem Ollama para o principal)."""
    settings = _dev_settings(
        llm_provider="llama_cpp",
        llm_model_path=str(tmp_path / "m.gguf"),
        utility_llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:4b",
    )
    settings.validate_runtime()  # não deve levantar
    assert settings.effective_utility_llm_provider == "ollama"


def test_utility_llama_cpp_without_path_falls_back_to_main(tmp_path) -> None:
    """UTILITY_LLM_PROVIDER=llama_cpp sem UTILITY_LLM_MODEL_PATH: válido,
    pois a factory faz o utilitário herdar o caminho do principal."""
    settings = _dev_settings(
        llm_provider="llama_cpp",
        llm_model_path=str(tmp_path / "m.gguf"),
        utility_llm_provider="llama_cpp",
    )
    settings.validate_runtime()  # não deve levantar


def test_utility_llama_cpp_separate_path_required_when_main_is_different_backend(tmp_path) -> None:
    """Principal ollama + utilitário llama_cpp: como não há GGUF do
    'principal' para herdar, UTILITY_LLM_MODEL_PATH passa a ser exigido."""
    settings = _dev_settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        utility_llm_provider="llama_cpp",
    )
    with pytest.raises(ConfigurationError, match="UTILITY_LLM_MODEL_PATH"):
        settings.validate_runtime()


# --- Comportamento anterior preservado (Claude/OpenAI) -------------------------


def test_claude_default_behavior_preserved() -> None:
    settings = _dev_settings(llm_provider="claude", anthropic_api_key="sk-fake")
    settings.validate_runtime()  # não deve levantar


def test_openai_default_behavior_preserved() -> None:
    settings = _dev_settings(llm_provider="openai", openai_api_key="sk-fake")
    settings.validate_runtime()


def test_invalid_llm_provider_name_fails() -> None:
    settings = _dev_settings(llm_provider="cohere")
    with pytest.raises(ConfigurationError, match="LLM_PROVIDER inválido"):
        settings.validate_runtime()


def test_invalid_utility_llm_provider_name_fails(tmp_path) -> None:
    settings = _dev_settings(
        llm_provider="llama_cpp",
        llm_model_path=str(tmp_path / "m.gguf"),
        utility_llm_provider="cohere",
    )
    with pytest.raises(ConfigurationError, match="UTILITY_LLM_PROVIDER inválido"):
        settings.validate_runtime()


# --- Embeddings ------------------------------------------------------------


def test_embedding_disabled_passes() -> None:
    settings = _dev_settings(embedding_provider="disabled")
    settings.validate_runtime()


def test_embedding_sentence_transformers_with_default_model_passes() -> None:
    settings = _dev_settings(embedding_provider="sentence_transformers")
    settings.validate_runtime()


def test_embedding_invalid_provider_name_fails() -> None:
    settings = _dev_settings(embedding_provider="pinecone")
    with pytest.raises(ConfigurationError, match="EMBEDDING_PROVIDER inválido"):
        settings.validate_runtime()

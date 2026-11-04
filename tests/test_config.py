import pytest

from src.config import Settings, load_settings

ENV_VARS = [
    "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_PATH", "QDRANT_COLLECTION", "EMBEDDING_MODEL",
    "RERANKER_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
    "BM25_K1", "BM25_B", "CANDIDATE_K", "FINAL_K", "RERANK_THRESHOLD", "LLM_TIMEOUT_S", "DEBUG",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_when_env_empty(clean_env):
    assert load_settings(env_file=None) == Settings()


def test_reads_and_casts(clean_env):
    clean_env.setenv("CHUNK_SIZE", "400")
    clean_env.setenv("RERANK_THRESHOLD", "0.55")
    clean_env.setenv("DEBUG", "true")
    s = load_settings(env_file=None)
    assert s.chunk_size == 400 and s.rerank_threshold == 0.55 and s.debug is True


def test_empty_string_is_none(clean_env):
    clean_env.setenv("QDRANT_PATH", "")
    clean_env.setenv("GEMINI_API_KEY", "")
    s = load_settings(env_file=None)
    assert s.qdrant_path is None and s.gemini_api_key is None


def test_invalid_number_raises_with_name(clean_env):
    clean_env.setenv("CHUNK_SIZE", "abc")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        load_settings(env_file=None)


def test_repr_hides_api_key():
    assert "secret-xyz" not in repr(Settings(gemini_api_key="secret-xyz"))

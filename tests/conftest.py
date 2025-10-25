import pytest
from unittest.mock import Mock, MagicMock
import json


@pytest.fixture
def mock_ollama_config():
    """Configuration par défaut pour OllamaProvider"""
    return {
        "api_url": "http://localhost:11434",
        "model": "llama3.2"
    }


@pytest.fixture
def mock_stream_chunks_ollama():
    """Chunks simulés pour le streaming Ollama"""
    return [
        {"response": "Hello", "done": False},
        {"response": " world", "done": False},
        {"response": "!", "done": False},
        {
            "response": "",
            "done": True,
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500000000,
            "eval_count": 50,
            "eval_duration": 2000000000,
            "context": [1, 2, 3],
            "done_reason": "stop",
            "model": "llama3.2",
            "created_at": "2025-01-15T10:00:00Z"
        }
    ]


@pytest.fixture
def mock_stream_chunks_ollama_chat():
    """Chunks simulés pour le chat streaming Ollama"""
    return [
        {"message": {"content": "Bonjour"}, "done": False},
        {"message": {"content": " !"}, "done": False},
        {
            "message": {"content": ""},
            "done": True,
            "total_duration": 3000000000,
            "eval_count": 30,
            "eval_duration": 1500000000,
            "model": "llama3.2"
        }
    ]


@pytest.fixture
def mock_stream_chunks_openai():
    """Chunks simulés pour le streaming OpenAI"""
    return [
        {
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
            "model": "gpt-4"
        },
        {
            "choices": [{"delta": {"content": " world"}, "finish_reason": None}],
            "model": "gpt-4"
        },
        {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    ]


@pytest.fixture
def mock_stream_chunks_anthropic():
    """Chunks simulés pour le streaming Anthropic"""
    return [
        {"type": "content_block_delta", "delta": {"text": "Hello"}},
        {"type": "content_block_delta", "delta": {"text": " Claude"}},
        {
            "type": "message_stop",
            "usage": {"input_tokens": 15, "output_tokens": 25},
            "model": "claude-3",
            "stop_reason": "end_turn"
        }
    ]


@pytest.fixture
def mock_download_chunks():
    """Chunks simulés pour le téléchargement Ollama"""
    return [
        {"status": "pulling manifest"},
        {
            "status": "pulling",
            "digest": "sha256:abcd1234efgh5678",
            "total": 1000000,
            "completed": 250000
        },
        {
            "status": "pulling",
            "digest": "sha256:abcd1234efgh5678",
            "total": 1000000,
            "completed": 500000
        },
        {
            "status": "pulling",
            "digest": "sha256:abcd1234efgh5678",
            "total": 1000000,
            "completed": 1000000
        },
        {"status": "verifying sha256 digest"},
        {"status": "writing manifest"},
        {"status": "removing any unused layers"},
        {"status": "success"}
    ]


@pytest.fixture
def mock_requests_response():
    """Mock générique pour requests.Response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.fixture
def mock_requests_stream_response(mock_stream_chunks_ollama):
    """Mock pour une réponse streaming"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    def iter_lines():
        for chunk in mock_stream_chunks_ollama:
            yield json.dumps(chunk).encode('utf-8')

    mock_response.iter_lines = iter_lines
    return mock_response
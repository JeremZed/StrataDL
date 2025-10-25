import pytest
from stratadl.core.provider.stream import (
    BaseStreamResult,
    OllamaStreamResult,
    OpenAIStreamResult,
    AnthropicStreamResult,
    OllamaDownloadResult
)


class TestOllamaStreamResult:
    """Tests pour OllamaStreamResult"""

    def test_stream_iteration(self, mock_stream_chunks_ollama):
        """Test l'itération sur le stream"""
        def generator():
            for chunk in mock_stream_chunks_ollama:
                yield chunk

        result = OllamaStreamResult(generator())
        chunks = list(result)

        assert len(chunks) == 4
        assert result.full_response == "Hello world!"
        assert result._consumed is True

    def test_extract_text_from_response(self, mock_stream_chunks_ollama):
        """Test l'extraction de texte depuis le champ 'response'"""
        def generator():
            yield mock_stream_chunks_ollama[0]

        result = OllamaStreamResult(generator())
        text = result._extract_text(mock_stream_chunks_ollama[0])
        assert text == "Hello"

    def test_extract_text_from_message(self, mock_stream_chunks_ollama_chat):
        """Test l'extraction de texte depuis le champ 'message'"""
        def generator():
            yield mock_stream_chunks_ollama_chat[0]

        result = OllamaStreamResult(generator())
        text = result._extract_text(mock_stream_chunks_ollama_chat[0])
        assert text == "Bonjour"

    def test_is_done(self, mock_stream_chunks_ollama):
        """Test la détection du dernier chunk"""
        result = OllamaStreamResult(iter([]))
        assert result._is_done(mock_stream_chunks_ollama[0]) is False
        assert result._is_done(mock_stream_chunks_ollama[-1]) is True

    def test_extract_metadata(self, mock_stream_chunks_ollama):
        """Test l'extraction des métadonnées"""
        result = OllamaStreamResult(iter([]))
        metadata = result._extract_metadata(mock_stream_chunks_ollama[-1])

        assert metadata['total_duration'] == 5000000000
        assert metadata['eval_count'] == 50
        assert metadata['model'] == "llama3.2"
        assert metadata['done_reason'] == "stop"

    def test_get_full_response_without_iteration(self, mock_stream_chunks_ollama):
        """Test get_full_response sans avoir itéré"""
        def generator():
            for chunk in mock_stream_chunks_ollama:
                yield chunk

        result = OllamaStreamResult(generator())
        full_response = result.get_full_response()

        assert full_response == "Hello world!"
        assert result._consumed is True

    def test_get_metadata_without_iteration(self, mock_stream_chunks_ollama):
        """Test get_metadata sans avoir itéré"""
        def generator():
            for chunk in mock_stream_chunks_ollama:
                yield chunk

        result = OllamaStreamResult(generator())
        metadata = result.get_metadata()

        assert metadata is not None
        assert metadata['model'] == "llama3.2"

    def test_get_stats(self, mock_stream_chunks_ollama):
        """Test le calcul des statistiques"""
        def generator():
            for chunk in mock_stream_chunks_ollama:
                yield chunk

        result = OllamaStreamResult(generator())
        stats = result.get_stats()

        assert stats['total_time_seconds'] == 5.0
        assert stats['generation_time'] == 2.0
        assert stats['tokens_generated'] == 50
        assert stats['tokens_per_second'] == 25.0
        assert stats['model'] == "llama3.2"

    def test_double_iteration(self, mock_stream_chunks_ollama):
        """Test une double itération (doit utiliser les chunks mis en cache)"""
        def generator():
            for chunk in mock_stream_chunks_ollama:
                yield chunk

        result = OllamaStreamResult(generator())

        # Première itération
        chunks1 = list(result)
        # Deuxième itération
        chunks2 = list(result)

        assert chunks1 == chunks2
        assert len(chunks2) == 4


class TestOpenAIStreamResult:
    """Tests pour OpenAIStreamResult"""

    def test_stream_iteration(self, mock_stream_chunks_openai):
        """Test l'itération sur le stream OpenAI"""
        def generator():
            for chunk in mock_stream_chunks_openai:
                yield chunk

        result = OpenAIStreamResult(generator())
        chunks = list(result)

        assert len(chunks) == 3
        assert result.full_response == "Hello world"

    def test_extract_text(self, mock_stream_chunks_openai):
        """Test l'extraction de texte"""
        result = OpenAIStreamResult(iter([]))
        text = result._extract_text(mock_stream_chunks_openai[0])
        assert text == "Hello"

    def test_extract_text_empty_delta(self):
        """Test l'extraction avec delta vide"""
        result = OpenAIStreamResult(iter([]))
        chunk = {"choices": [{"delta": {}}]}
        text = result._extract_text(chunk)
        assert text == ""

    def test_is_done(self, mock_stream_chunks_openai):
        """Test la détection du dernier chunk"""
        result = OpenAIStreamResult(iter([]))
        assert result._is_done(mock_stream_chunks_openai[0]) is False
        assert result._is_done(mock_stream_chunks_openai[-1]) is True

    def test_extract_metadata(self, mock_stream_chunks_openai):
        """Test l'extraction des métadonnées"""
        result = OpenAIStreamResult(iter([]))
        metadata = result._extract_metadata(mock_stream_chunks_openai[-1])

        assert metadata['usage']['prompt_tokens'] == 10
        assert metadata['usage']['completion_tokens'] == 20
        assert metadata['model'] == "gpt-4"
        assert metadata['finish_reason'] == "stop"

    def test_get_stats(self, mock_stream_chunks_openai):
        """Test le calcul des statistiques"""
        def generator():
            for chunk in mock_stream_chunks_openai:
                yield chunk

        result = OpenAIStreamResult(generator())
        stats = result.get_stats()

        assert stats['prompt_tokens'] == 10
        assert stats['completion_tokens'] == 20
        assert stats['total_tokens'] == 30
        assert stats['model'] == "gpt-4"
        assert stats['finish_reason'] == "stop"


class TestAnthropicStreamResult:
    """Tests pour AnthropicStreamResult"""

    def test_stream_iteration(self, mock_stream_chunks_anthropic):
        """Test l'itération sur le stream Anthropic"""
        def generator():
            for chunk in mock_stream_chunks_anthropic:
                yield chunk

        result = AnthropicStreamResult(generator())
        chunks = list(result)

        assert len(chunks) == 3
        assert result.full_response == "Hello Claude"

    def test_extract_text(self, mock_stream_chunks_anthropic):
        """Test l'extraction de texte"""
        result = AnthropicStreamResult(iter([]))
        text = result._extract_text(mock_stream_chunks_anthropic[0])
        assert text == "Hello"

    def test_extract_text_wrong_type(self):
        """Test l'extraction avec un type différent"""
        result = AnthropicStreamResult(iter([]))
        chunk = {"type": "other_type", "delta": {"text": "ignored"}}
        text = result._extract_text(chunk)
        assert text == ""

    def test_is_done(self, mock_stream_chunks_anthropic):
        """Test la détection du dernier chunk"""
        result = AnthropicStreamResult(iter([]))
        assert result._is_done(mock_stream_chunks_anthropic[0]) is False
        assert result._is_done(mock_stream_chunks_anthropic[-1]) is True

    def test_extract_metadata(self, mock_stream_chunks_anthropic):
        """Test l'extraction des métadonnées"""
        result = AnthropicStreamResult(iter([]))
        metadata = result._extract_metadata(mock_stream_chunks_anthropic[-1])

        assert metadata['usage']['input_tokens'] == 15
        assert metadata['usage']['output_tokens'] == 25
        assert metadata['model'] == "claude-3"
        assert metadata['stop_reason'] == "end_turn"

    def test_get_stats(self, mock_stream_chunks_anthropic):
        """Test le calcul des statistiques"""
        def generator():
            for chunk in mock_stream_chunks_anthropic:
                yield chunk

        result = AnthropicStreamResult(generator())
        stats = result.get_stats()

        assert stats['input_tokens'] == 15
        assert stats['output_tokens'] == 25
        assert stats['model'] == "claude-3"
        assert stats['stop_reason'] == "end_turn"


class TestOllamaDownloadResult:
    """Tests pour OllamaDownloadResult"""

    def test_download_iteration(self, mock_download_chunks):
        """Test l'itération sur le téléchargement"""
        def generator():
            for chunk in mock_download_chunks:
                yield chunk

        result = OllamaDownloadResult(generator())
        chunks = list(result)

        assert len(chunks) == 8
        assert result.current_status == "success"
        assert result._consumed is True

    def test_extract_text(self, mock_download_chunks):
        """Test l'extraction du statut"""
        result = OllamaDownloadResult(iter([]))
        text = result._extract_text(mock_download_chunks[0])
        assert text == "pulling manifest"

    def test_is_done(self, mock_download_chunks):
        """Test la détection de la fin du téléchargement"""
        result = OllamaDownloadResult(iter([]))
        assert result._is_done(mock_download_chunks[0]) is False
        assert result._is_done(mock_download_chunks[-1]) is True

    def test_progress_tracking(self, mock_download_chunks):
        """Test le suivi de la progression"""
        def generator():
            for chunk in mock_download_chunks:
                yield chunk

        result = OllamaDownloadResult(generator())
        list(result)  # Consommer le générateur

        assert result.total_size == 1000000
        assert result.downloaded_size == 1000000
        assert len(result.download_progress) == 3

    def test_get_progress_percentage(self, mock_download_chunks):
        """Test le calcul du pourcentage"""
        def generator():
            for chunk in mock_download_chunks[:4]:  # Jusqu'à 50%
                yield chunk

        result = OllamaDownloadResult(generator())
        list(result)

        percentage = result.get_progress_percentage()
        assert percentage == 100.0  # Dernier chunk est à 100%

    def test_get_progress_percentage_zero_total(self):
        """Test du pourcentage quand total_size est 0"""
        result = OllamaDownloadResult(iter([]))
        percentage = result.get_progress_percentage()
        assert percentage == 0

    def test_get_stats(self, mock_download_chunks):
        """Test le calcul des statistiques"""
        def generator():
            for chunk in mock_download_chunks:
                yield chunk

        result = OllamaDownloadResult(generator())
        stats = result.get_stats()

        assert stats['status'] == "success"
        assert stats['total_size_bytes'] == 1000000
        assert stats['total_size_mb'] == pytest.approx(0.95, abs=0.1)
        assert stats['num_progress_updates'] == 3

    def test_extract_metadata(self, mock_download_chunks):
        """Test l'extraction des métadonnées finales"""
        def generator():
            for chunk in mock_download_chunks:
                yield chunk

        result = OllamaDownloadResult(generator())
        list(result)

        metadata = result._extract_metadata(mock_download_chunks[-1])
        assert metadata['status'] == "success"
        assert metadata['total_size'] == 1000000
        assert len(metadata['progress_history']) == 3
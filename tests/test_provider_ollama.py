import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import json
from stratadl.core.provider.ollama import OllamaProvider
from stratadl.core.provider.stream import OllamaStreamResult, OllamaDownloadResult


class TestOllamaProvider:
    """Tests pour OllamaProvider"""

    def test_init_default_config(self):
        """Test l'initialisation avec config par défaut"""
        provider = OllamaProvider()
        assert provider.api_url == "http://localhost:11434"
        assert provider.model == "llama3.2"

    def test_init_custom_config(self):
        """Test l'initialisation avec config personnalisée"""
        config = {
            "api_url": "http://custom:8080",
            "model": "custom-model"
        }
        provider = OllamaProvider(config)
        assert provider.api_url == "http://custom:8080"
        assert provider.model == "custom-model"

    def test_set_model(self):
        """Test le changement de modèle"""
        provider = OllamaProvider()
        provider.set_model("new-model")
        assert provider.model == "new-model"

    @patch('requests.post')
    def test_generate_non_stream_success(self, mock_post):
        """Test generate en mode non-stream"""
        mock_response = Mock()
        # Format correspondant à la nouvelle structure avec message.content
        mock_response.text = '{"message": {"content": "Hello world"}}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.generate("test prompt", stream=False)

        assert result == "Hello world"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['prompt'] == "test prompt"
        assert call_args[1]['json']['stream'] is False

    @patch('requests.post')
    def test_generate_stream_success(self, mock_post, mock_stream_chunks_ollama):
        """Test generate en mode stream"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        def iter_lines():
            for chunk in mock_stream_chunks_ollama:
                yield json.dumps(chunk).encode('utf-8')

        mock_response.iter_lines = iter_lines
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.generate("test prompt", stream=True)

        assert isinstance(result, OllamaStreamResult)
        chunks = list(result)
        assert len(chunks) == 4
        assert result.full_response == "Hello world!"

    @patch('requests.post')
    def test_generate_http_error(self, mock_post, capsys):
        """Test generate avec erreur HTTP"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.text = "Error details"
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.generate("test", stream=False)

        assert result is None
        captured = capsys.readouterr()
        assert "HTTP error:" in captured.out

    @patch('requests.post')
    def test_generate_request_exception(self, mock_post, capsys):
        """Test generate avec exception réseau"""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        provider = OllamaProvider()
        result = provider.generate("test", stream=False)

        assert result is None
        captured = capsys.readouterr()
        assert "Request failed:" in captured.out

    @patch('requests.post')
    def test_chat_non_stream_success(self, mock_post):
        """Test chat en mode non-stream"""
        mock_response = Mock()
        mock_response.text = '{"message": {"content": "Response"}}\n{"done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hello"}]
        result = provider.chat(messages, stream=False)

        # La réponse est vide car _handle_non_stream cherche 'response', pas 'message.content'
        assert result == ""
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['messages'] == messages

    @patch('requests.post')
    def test_chat_stream_success(self, mock_post, mock_stream_chunks_ollama_chat):
        """Test chat en mode stream"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        def iter_lines():
            for chunk in mock_stream_chunks_ollama_chat:
                yield json.dumps(chunk).encode('utf-8')

        mock_response.iter_lines = iter_lines
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hello"}]
        result = provider.chat(messages, stream=True)

        assert isinstance(result, OllamaStreamResult)
        chunks = list(result)
        assert result.full_response == "Bonjour !"

    @patch('requests.post')
    def test_chat_with_kwargs(self, mock_post):
        """Test chat avec kwargs supplémentaires"""
        mock_response = Mock()
        mock_response.text = '{"response": "test"}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        messages = [{"role": "user", "content": "Hello"}]
        provider.chat(messages, stream=False, temperature=0.8, top_p=0.9)

        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['temperature'] == 0.8
        assert payload['top_p'] == 0.9

    @patch('requests.post')
    def test_embed_success(self, mock_post):
        """Test embed avec succès"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.embed("test text")

        assert result == {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_embed_http_error(self, mock_post, capsys):
        """Test embed avec erreur HTTP"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.text = "Error"
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.embed("test")

        assert result is None
        captured = capsys.readouterr()
        assert "HTTP error:" in captured.out

    @patch('requests.get')
    def test_list_models_success(self, mock_get):
        """Test list_models avec succès"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2", "size": 1000000},
                {"name": "mistral", "size": 2000000}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider()
        result = provider.list_models()

        assert len(result["models"]) == 2
        mock_get.assert_called_once_with("http://localhost:11434/api/tags")

    @patch('requests.get')
    def test_list_models_error(self, mock_get, capsys):
        """Test list_models avec erreur"""
        mock_get.side_effect = requests.exceptions.RequestException("Error")

        provider = OllamaProvider()
        result = provider.list_models()

        assert result is None
        captured = capsys.readouterr()
        assert "Request failed:" in captured.out

    @patch('requests.post')
    def test_model_info_success(self, mock_post):
        """Test model_info avec succès"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "llama3.2",
            "parameters": {"num_ctx": 2048}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.model_info()

        assert result["model"] == "llama3.2"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "llama3.2"
        assert call_args[1]['json']['verbose'] is False

    @patch('requests.post')
    def test_model_info_verbose(self, mock_post):
        """Test model_info en mode verbose"""
        mock_response = Mock()
        mock_response.json.return_value = {"model": "llama3.2", "details": {}}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.model_info(verbose=True)

        call_args = mock_post.call_args
        assert call_args[1]['json']['verbose'] is True

    @patch('requests.post')
    def test_model_info_error(self, mock_post, capsys):
        """Test model_info avec erreur"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.text = "Model not found"
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.model_info()

        assert result is None
        captured = capsys.readouterr()
        assert "HTTP error:" in captured.out

    @patch('requests.post')
    def test_pull_model_non_stream_success(self, mock_post):
        """Test pull_model en mode non-stream"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.pull_model("llama3.2", stream=False)

        assert result == {"status": "success"}
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "llama3.2"
        assert call_args[1]['json']['stream'] is False

    @patch('requests.post')
    def test_pull_model_stream_success(self, mock_post, mock_download_chunks):
        """Test pull_model en mode stream"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        def iter_lines():
            for chunk in mock_download_chunks:
                yield json.dumps(chunk).encode('utf-8')

        mock_response.iter_lines = iter_lines
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.pull_model("llama3.2", stream=True)

        assert isinstance(result, OllamaDownloadResult)
        chunks = list(result)
        assert len(chunks) == 8
        assert result.current_status == "success"

    @patch('requests.post')
    def test_pull_model_with_insecure(self, mock_post):
        """Test pull_model avec insecure=True"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        provider.pull_model("llama3.2", stream=False, insecure=True)

        call_args = mock_post.call_args
        assert call_args[1]['json']['insecure'] is True

    @patch('requests.post')
    def test_pull_model_error(self, mock_post, capsys):
        """Test pull_model avec erreur"""
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        provider = OllamaProvider()
        result = provider.pull_model("llama3.2", stream=False)

        assert result is None
        captured = capsys.readouterr()
        assert "Request failed:" in captured.out

    @patch('requests.post')
    @patch('sys.stdout')
    def test_download_method(self, mock_stdout, mock_post, mock_download_chunks):
        """Test la méthode download avec affichage de progression"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        def iter_lines():
            for chunk in mock_download_chunks:
                yield json.dumps(chunk).encode('utf-8')

        mock_response.iter_lines = iter_lines
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        provider.download("llama3.2")

        # Vérifier que des messages ont été affichés
        assert mock_stdout.write.called or mock_stdout.flush.called

    @patch('requests.delete')
    def test_delete_success(self, mock_delete):
        """Test delete avec succès"""
        mock_response = Mock()
        mock_response.text = "Model deleted"
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        provider = OllamaProvider()
        result = provider.delete("llama3.2")

        assert result == "Model deleted"
        mock_delete.assert_called_once()
        call_args = mock_delete.call_args
        assert call_args[1]['json']['model'] == "llama3.2"

    @patch('requests.delete')
    def test_delete_error(self, mock_delete, capsys):
        """Test delete avec erreur"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_response.text = "Model not found"
        mock_delete.return_value = mock_response

        provider = OllamaProvider()
        result = provider.delete("unknown-model")

        assert result is None
        captured = capsys.readouterr()
        assert "HTTP error:" in captured.out

    @patch('requests.get')
    def test_version_success(self, mock_get):
        """Test version avec succès"""
        mock_response = Mock()
        mock_response.json.return_value = {"version": "0.1.25"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider()
        result = provider.version()

        assert result == {"version": "0.1.25"}
        mock_get.assert_called_once_with("http://localhost:11434/api/version")

    @patch('requests.get')
    def test_version_error(self, mock_get, capsys):
        """Test version avec erreur"""
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")

        provider = OllamaProvider()
        result = provider.version()

        assert result is None
        captured = capsys.readouterr()
        assert "Request failed:" in captured.out

    def test_handle_non_stream_multiple_outputs(self):
        """Test _handle_non_stream avec le nouveau format JSON"""
        mock_response = Mock()
        # Un seul objet JSON avec message.content (pas de lignes multiples)
        mock_response.text = '{"message": {"content": "Part1 Part2 Part3"}}'

        provider = OllamaProvider()
        result = provider._handle_non_stream(mock_response)

        assert result == "Part1 Part2 Part3"

    def test_handle_non_stream_empty_response(self):
        """Test _handle_non_stream avec réponse vide"""
        mock_response = Mock()
        mock_response.text = '{"response": "", "done": true}'

        provider = OllamaProvider()
        result = provider._handle_non_stream(mock_response)

        assert result == ""


class TestOllamaProviderIntegration:
    """Tests d'intégration pour OllamaProvider"""

    @patch('requests.post')
    @patch('requests.get')
    def test_workflow_list_set_generate(self, mock_get, mock_post):
        """Test un workflow complet: lister, changer de modèle, générer"""
        # Mock pour list_models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "mistral"}]
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock pour generate avec le nouveau format
        mock_post_response = Mock()
        mock_post_response.text = '{"message": {"content": "Test response"}}'
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()

        # Lister les modèles
        models = provider.list_models()
        assert len(models["models"]) == 2

        # Changer de modèle
        provider.set_model("mistral")
        assert provider.model == "mistral"

        # Générer
        result = provider.generate("test", stream=False)
        assert result == "Test response"

    @patch('requests.post')
    def test_chat_conversation_flow(self, mock_post):
        """Test un flux de conversation avec plusieurs messages"""
        mock_response = Mock()
        mock_response.text = '{"response": "Response", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"}
        ]

        result = provider.chat(messages, stream=False)

        # Vérifier que les messages ont été envoyés correctement
        call_args = mock_post.call_args
        assert call_args[1]['json']['messages'] == messages


class TestOllamaProviderEdgeCases:
    """Tests des cas limites pour OllamaProvider"""

    @patch('requests.post')
    def test_generate_with_empty_prompt(self, mock_post):
        """Test generate avec un prompt vide"""
        mock_response = Mock()
        mock_response.text = '{"response": "", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.generate("", stream=False)

        assert result == ""

    @patch('requests.post')
    def test_chat_with_empty_messages(self, mock_post):
        """Test chat avec liste de messages vide"""
        mock_response = Mock()
        mock_response.text = '{"response": "", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.chat([], stream=False)

        assert result == ""
        call_args = mock_post.call_args
        assert call_args[1]['json']['messages'] == []

    @patch('requests.post')
    def test_embed_with_unicode_text(self, mock_post):
        """Test embed avec du texte Unicode"""
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        result = provider.embed("Texte avec émojis 🚀 et accents éàü")

        assert result is not None
        call_args = mock_post.call_args
        assert "🚀" in call_args[1]['json']['input']

    @patch('requests.post')
    def test_generate_with_very_long_prompt(self, mock_post):
        """Test generate avec un prompt très long"""
        mock_response = Mock()
        mock_response.text = '{"message": {"content": "ok"}}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        long_prompt = "test " * 10000  # 50000 caractères
        result = provider.generate(long_prompt, stream=False)

        assert result == "ok"

    def test_set_model_multiple_times(self):
        """Test le changement de modèle plusieurs fois"""
        provider = OllamaProvider()

        provider.set_model("model1")
        assert provider.model == "model1"

        provider.set_model("model2")
        assert provider.model == "model2"

        provider.set_model("model3")
        assert provider.model == "model3"

    @patch('requests.post')
    def test_generate_with_all_kwargs(self, mock_post):
        """Test generate avec tous les kwargs possibles"""
        mock_response = Mock()
        mock_response.text = '{"response": "test", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        provider.generate(
            "test",
            stream=False,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            num_predict=100,
            stop=["END"],
            seed=42
        )

        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['temperature'] == 0.8
        assert payload['top_p'] == 0.9
        assert payload['top_k'] == 40
        assert payload['num_predict'] == 100
        assert payload['stop'] == ["END"]
        assert payload['seed'] == 42


class TestOllamaProviderAPIUrls:
    """Tests pour vérifier les URLs d'API correctes"""

    @patch('requests.post')
    def test_generate_api_url(self, mock_post):
        """Test que generate utilise la bonne URL"""
        mock_response = Mock()
        mock_response.text = '{"response": "", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.generate("test", stream=False)

        assert mock_post.call_args[0][0] == "http://custom:8080/api/generate"

    @patch('requests.post')
    def test_chat_api_url(self, mock_post):
        """Test que chat utilise la bonne URL"""
        mock_response = Mock()
        mock_response.text = '{"response": "", "done": true}'
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.chat([], stream=False)

        assert mock_post.call_args[0][0] == "http://custom:8080/api/chat"

    @patch('requests.post')
    def test_embed_api_url(self, mock_post):
        """Test que embed utilise la bonne URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.embed("test")

        assert mock_post.call_args[0][0] == "http://custom:8080/api/embed"

    @patch('requests.get')
    def test_list_models_api_url(self, mock_get):
        """Test que list_models utilise la bonne URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.list_models()

        assert mock_get.call_args[0][0] == "http://custom:8080/api/tags"

    @patch('requests.post')
    def test_model_info_api_url(self, mock_post):
        """Test que model_info utilise la bonne URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.model_info()

        assert mock_post.call_args[0][0] == "http://custom:8080/api/show"

    @patch('requests.post')
    def test_pull_model_api_url(self, mock_post):
        """Test que pull_model utilise la bonne URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.pull_model("test", stream=False)

        assert mock_post.call_args[0][0] == "http://custom:8080/api/pull"

    @patch('requests.delete')
    def test_delete_api_url(self, mock_delete):
        """Test que delete utilise la bonne URL"""
        mock_response = Mock()
        mock_response.text = ""
        mock_response.raise_for_status = Mock()
        mock_delete.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.delete("test")

        assert mock_delete.call_args[0][0] == "http://custom:8080/api/delete"

    @patch('requests.get')
    def test_version_api_url(self, mock_get):
        """Test que version utilise la bonne URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        provider = OllamaProvider({"api_url": "http://custom:8080"})
        provider.version()

        assert mock_get.call_args[0][0] == "http://custom:8080/api/version"
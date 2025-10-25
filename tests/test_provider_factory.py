import pytest
from stratadl.core.provider.factory import get_provider
from stratadl.core.provider.ollama import OllamaProvider


class TestProviderFactory:
    """Tests pour la factory de providers"""

    def test_get_ollama_provider_default(self):
        """Test la création d'un provider Ollama sans config"""
        provider = get_provider("ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.api_url == "http://localhost:11434"
        assert provider.model == "llama3.2"

    def test_get_ollama_provider_with_config(self, mock_ollama_config):
        """Test la création d'un provider Ollama avec config"""
        config = {
            "api_url": "http://custom:8080",
            "model": "custom-model"
        }
        provider = get_provider("ollama", config=config)
        assert isinstance(provider, OllamaProvider)
        assert provider.api_url == "http://custom:8080"
        assert provider.model == "custom-model"

    def test_get_provider_invalid_name(self):
        """Test avec un nom de provider invalide"""
        with pytest.raises(ValueError) as exc_info:
            get_provider("invalid_provider")
        assert "Provider 'invalid_provider' inconnu" in str(exc_info.value)

    def test_get_provider_case_sensitive(self):
        """Test que le nom du provider est sensible à la casse"""
        with pytest.raises(ValueError):
            get_provider("OLLAMA")

    def test_get_provider_with_kwargs(self):
        """Test la création d'un provider avec kwargs supplémentaires"""
        provider = get_provider("ollama", config={"model": "test"})
        assert isinstance(provider, OllamaProvider)
import pytest
from stratadl.core.provider.base import BaseProvider


class ConcreteProvider(BaseProvider):
    """Implémentation concrète pour les tests"""

    def generate(self, prompt: str, stream: bool, **kwargs) -> str:
        return f"Generated: {prompt}"

    def chat(self, prompt: str, stream: bool, **kwargs) -> str:
        return f"Chat: {prompt}"

    def embed(self, text: str, **kwargs) -> list:
        return [0.1, 0.2, 0.3]


class TestBaseProvider:
    """Tests pour la classe BaseProvider"""

    def test_init_without_config(self):
        """Test l'initialisation sans configuration"""
        provider = ConcreteProvider()
        assert provider.config == {}
        assert provider.provider_name == "concrete"

    def test_init_with_config(self):
        """Test l'initialisation avec configuration"""
        config = {"model": "test-model", "api_key": "test-key"}
        provider = ConcreteProvider(config)
        assert provider.config == config
        assert provider.provider_name == "concrete"

    def test_provider_name_extraction(self):
        """Test l'extraction du nom du provider"""
        provider = ConcreteProvider()
        assert provider.provider_name == "concrete"

    def test_abstract_methods_implemented(self):
        """Test que les méthodes abstraites sont implémentées"""
        provider = ConcreteProvider()
        assert callable(provider.generate)
        assert callable(provider.chat)
        assert callable(provider.embed)

    def test_cannot_instantiate_base_provider(self):
        """Test qu'on ne peut pas instancier BaseProvider directement"""
        with pytest.raises(TypeError):
            BaseProvider()

    def test_generate_method(self):
        """Test la méthode generate"""
        provider = ConcreteProvider()
        result = provider.generate("test prompt", stream=False)
        assert result == "Generated: test prompt"

    def test_chat_method(self):
        """Test la méthode chat"""
        provider = ConcreteProvider()
        result = provider.chat("test message", stream=False)
        assert result == "Chat: test message"

    def test_embed_method(self):
        """Test la méthode embed"""
        provider = ConcreteProvider()
        result = provider.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 3
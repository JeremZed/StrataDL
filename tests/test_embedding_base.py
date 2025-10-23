import pytest
import numpy as np
from abc import ABC
from stratadl.core.embedding.base import BaseEmbeddingModel


class DummyEmbeddingModel(BaseEmbeddingModel):
    """Implémentation factice pour tester la classe abstraite."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def encode(self, texts, batch_size=32, show_progress=False, normalize=False):
        texts = self._validate_texts(texts)
        embeddings = np.random.randn(len(texts), self._dim).astype(np.float32)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    @property
    def embedding_dim(self):
        return self._dim


class TestBaseEmbeddingModel:
    """Tests pour la classe abstraite BaseEmbeddingModel."""

    def test_cannot_instantiate_abstract_class(self):
        """La classe abstraite ne peut pas être instanciée directement."""
        with pytest.raises(TypeError):
            BaseEmbeddingModel()

    def test_validate_texts_single_string(self):
        """_validate_texts doit convertir un string en liste."""
        model = DummyEmbeddingModel()
        result = model._validate_texts("Hello world")
        assert result == ["Hello world"]
        assert isinstance(result, list)

    def test_validate_texts_list(self):
        """_validate_texts doit retourner la liste telle quelle."""
        model = DummyEmbeddingModel()
        texts = ["Hello", "World"]
        result = model._validate_texts(texts)
        assert result == texts

    def test_validate_texts_empty_list_raises_error(self):
        """_validate_texts doit lever une erreur pour une liste vide."""
        model = DummyEmbeddingModel()
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            model._validate_texts([])

    def test_encode_returns_correct_shape(self):
        """encode doit retourner un array de la bonne forme."""
        model = DummyEmbeddingModel(dim=128)
        embeddings = model.encode(["text1", "text2", "text3"])
        assert embeddings.shape == (3, 128)
        assert embeddings.dtype == np.float32

    def test_normalize_option(self):
        """L'option normalize doit normaliser les embeddings."""
        model = DummyEmbeddingModel()
        embeddings = model.encode(["text"], normalize=True)
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-6)
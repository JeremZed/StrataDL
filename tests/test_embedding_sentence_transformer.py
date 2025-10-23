import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from stratadl.core.embedding.sentence_transformer import (
    SentenceTransformerEmbedding,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


class TestSentenceTransformerEmbedding:
    """Tests pour SentenceTransformerEmbedding."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock de SentenceTransformer."""
        mock = Mock()
        mock.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock.get_sentence_embedding_dimension.return_value = 384
        return mock

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    def test_import_error_when_not_available(self):
        """Test de l'erreur d'import si sentence-transformers n'est pas disponible."""
        with pytest.raises(ImportError, match="sentence-transformers n'est pas installé"):
            SentenceTransformerEmbedding()

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_initialization_success(self, mock_st_class):
        """Test de l'initialisation réussie."""
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding(model_name="test-model", device="cpu")

        assert embedder.model_name == "test-model"
        mock_st_class.assert_called_once_with("test-model", device="cpu")

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_initialization_failure(self, mock_st_class):
        """Test de l'échec de l'initialisation."""
        mock_st_class.side_effect = Exception("Model not found")

        with pytest.raises(RuntimeError, match="Erreur lors du chargement"):
            SentenceTransformerEmbedding(model_name="invalid-model")

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_single_text(self, mock_st_class):
        """Test de l'encodage d'un seul texte."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding()
        result = embedder.encode("Hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once()

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_multiple_texts(self, mock_st_class):
        """Test de l'encodage de plusieurs textes."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding()
        texts = ["text1", "text2", "text3"]
        result = embedder.encode(texts)

        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_with_parameters(self, mock_st_class):
        """Test de l'encodage avec paramètres personnalisés."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding()
        embedder.encode(
            ["text1", "text2"],
            batch_size=16,
            show_progress=True,
            normalize=True
        )

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 16
        assert call_kwargs["show_progress_bar"] is True
        assert call_kwargs["normalize_embeddings"] is True
        assert call_kwargs["convert_to_numpy"] is True

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_error_handling(self, mock_st_class):
        """Test de la gestion des erreurs pendant l'encodage."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding()

        with pytest.raises(RuntimeError, match="Erreur lors de l'encoding"):
            embedder.encode("text")

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_embedding_dim_property(self, mock_st_class):
        """Test de la propriété embedding_dim."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding()
        assert embedder.embedding_dim == 512

    @patch("stratadl.core.embedding.sentence_transformer.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("stratadl.core.embedding.sentence_transformer.SentenceTransformer")
    def test_repr(self, mock_st_class):
        """Test de la méthode __repr__."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedding(model_name="test-model")
        repr_str = repr(embedder)

        assert "SentenceTransformerEmbedding" in repr_str
        assert "test-model" in repr_str
        assert "384" in repr_str
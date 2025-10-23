import pytest
from unittest.mock import patch, Mock
from stratadl.core.embedding.factory import get_embedding_model
from stratadl.core.embedding.base import BaseEmbeddingModel
from stratadl.core.embedding.huggingface import HuggingFaceEmbedding
from stratadl.core.embedding.sentence_transformer import SentenceTransformerEmbedding


class TestGetEmbeddingModel:
    """Tests pour la factory get_embedding_model."""

    @patch("stratadl.core.embedding.factory.HuggingFaceEmbedding")
    def test_get_huggingface_model_default(self, mock_hf):
        """Test de création d'un modèle HuggingFace avec paramètres par défaut."""
        mock_instance = Mock(spec=HuggingFaceEmbedding)
        mock_hf.return_value = mock_instance

        model = get_embedding_model("huggingface")

        mock_hf.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=None
        )
        assert model is mock_instance

    @patch("stratadl.core.embedding.factory.HuggingFaceEmbedding")
    def test_get_huggingface_model_custom(self, mock_hf):
        """Test de création d'un modèle HuggingFace avec paramètres personnalisés."""
        mock_instance = Mock(spec=HuggingFaceEmbedding)
        mock_hf.return_value = mock_instance

        model = get_embedding_model(
            "huggingface",
            model_name="bert-base-uncased",
            device="cuda",
            max_length=256
        )

        mock_hf.assert_called_once_with(
            model_name="bert-base-uncased",
            device="cuda",
            max_length=256
        )

    @patch("stratadl.core.embedding.factory.SentenceTransformerEmbedding")
    def test_get_sentence_transformer_default(self, mock_st):
        """Test de création d'un modèle SentenceTransformer par défaut."""
        mock_instance = Mock(spec=SentenceTransformerEmbedding)
        mock_st.return_value = mock_instance

        model = get_embedding_model("sentence_transformer")

        mock_st.assert_called_once_with(
            model_name="all-MiniLM-L6-v2",
            device=None
        )

    @patch("stratadl.core.embedding.factory.SentenceTransformerEmbedding")
    def test_get_sentence_transformer_custom(self, mock_st):
        """Test de création d'un modèle SentenceTransformer personnalisé."""
        mock_instance = Mock(spec=SentenceTransformerEmbedding)
        mock_st.return_value = mock_instance

        model = get_embedding_model(
            "sentence_transformer",
            model_name="paraphrase-MiniLM-L6-v2",
            device="cpu"
        )

        mock_st.assert_called_once_with(
            model_name="paraphrase-MiniLM-L6-v2",
            device="cpu"
        )

    def test_invalid_backend_raises_error(self):
        """Test d'erreur pour un backend invalide."""
        with pytest.raises(ValueError, match="Backend 'invalid' inconnu"):
            get_embedding_model("invalid")

    @patch("stratadl.core.embedding.factory.SentenceTransformerEmbedding")
    def test_default_backend_is_sentence_transformer(self, mock_st):
        """Test que le backend par défaut est sentence_transformer."""
        mock_instance = Mock(spec=SentenceTransformerEmbedding)
        mock_st.return_value = mock_instance

        model = get_embedding_model()

        mock_st.assert_called_once()

    @patch("stratadl.core.embedding.factory.HuggingFaceEmbedding")
    def test_model_name_override(self, mock_hf):
        """Test que model_name override le modèle par défaut."""
        mock_instance = Mock(spec=HuggingFaceEmbedding)
        mock_hf.return_value = mock_instance

        custom_name = "custom-bert-model"
        model = get_embedding_model("huggingface", model_name=custom_name)

        call_kwargs = mock_hf.call_args[1]
        assert call_kwargs["model_name"] == custom_name
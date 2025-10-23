import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from stratadl.core.embedding.huggingface import HuggingFaceEmbedding


class TestHuggingFaceEmbedding:
    """Tests pour HuggingFaceEmbedding."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock du tokenizer HuggingFace."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones((2, 10), dtype=torch.long),
        }
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Mock du modèle HuggingFace."""
        model = Mock()
        model.config.hidden_size = 384
        model.eval.return_value = model
        model.to.return_value = model

        # Mock de la sortie du modèle
        output = Mock()
        output.last_hidden_state = torch.randn(2, 10, 384)
        model.return_value = output

        return model

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_initialization_success(self, mock_tokenizer_class, mock_model_class):
        """Test de l'initialisation réussie du modèle."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 384
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding(model_name="test-model")

        assert embedder.model_name == "test-model"
        assert embedder.max_length == 512
        assert embedder.device in ["cuda", "cpu"]
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with("test-model")

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_initialization_failure(self, mock_tokenizer_class, mock_model_class):
        """Test de l'échec de l'initialisation."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Erreur lors du chargement"):
            HuggingFaceEmbedding(model_name="invalid-model")

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    @patch("stratadl.core.embedding.huggingface.torch.cuda.is_available")
    def test_device_selection_cuda(self, mock_cuda, mock_tokenizer_class, mock_model_class):
        """Test de la sélection automatique du device CUDA."""
        mock_cuda.return_value = True
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 384
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding()
        assert embedder.device == "cuda"

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    @patch("stratadl.core.embedding.huggingface.torch.cuda.is_available")
    def test_device_selection_cpu(self, mock_cuda, mock_tokenizer_class, mock_model_class):
        """Test de la sélection automatique du device CPU."""
        mock_cuda.return_value = False
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model.config.hidden_size = 384
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding()
        assert embedder.device == "cpu"

    def test_mean_pooling(self):
        """Test de la fonction mean pooling."""
        embedder = HuggingFaceEmbedding.__new__(HuggingFaceEmbedding)

        # Créer des données de test
        batch_size, seq_len, hidden_size = 2, 5, 384
        model_output = Mock()
        model_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0]
        ])

        result = embedder._mean_pooling(model_output, attention_mask)

        assert result.shape == (batch_size, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_encode_single_text(self, mock_tokenizer_class, mock_model_class):
        """Test de l'encodage d'un seul texte."""
        # Setup mocks
        tokenizer = Mock()

        def tokenizer_call(*args, **kwargs):
            # Retourner des tensors sur CPU
            return {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones((1, 10), dtype=torch.long),
            }

        tokenizer.side_effect = tokenizer_call
        mock_tokenizer_class.from_pretrained.return_value = tokenizer

        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model

        def model_call(**kwargs):
            # Retourner des tensors sur CPU
            output = Mock()
            output.last_hidden_state = torch.randn(1, 10, 384)
            return output

        model.side_effect = model_call
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding(device="cpu")
        result = embedder.encode("Hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)
        assert result.dtype == np.float32

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_encode_multiple_texts(self, mock_tokenizer_class, mock_model_class):
        """Test de l'encodage de plusieurs textes."""
        tokenizer = Mock()

        def tokenizer_call(*args, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (3, 10)),
                "attention_mask": torch.ones((3, 10), dtype=torch.long),
            }

        tokenizer.side_effect = tokenizer_call
        mock_tokenizer_class.from_pretrained.return_value = tokenizer

        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model

        def model_call(**kwargs):
            output = Mock()
            output.last_hidden_state = torch.randn(3, 10, 384)
            return output

        model.side_effect = model_call
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding(device="cpu")
        texts = ["text1", "text2", "text3"]
        result = embedder.encode(texts)

        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_encode_with_batches(self, mock_tokenizer_class, mock_model_class):
        """Test de l'encodage avec plusieurs batchs."""
        tokenizer = Mock()

        def tokenizer_side_effect(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones((batch_size, 10), dtype=torch.long),
            }

        tokenizer.side_effect = tokenizer_side_effect
        mock_tokenizer_class.from_pretrained.return_value = tokenizer

        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model

        def model_side_effect(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            output = Mock()
            output.last_hidden_state = torch.randn(batch_size, 10, 384)
            return output

        model.side_effect = model_side_effect
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding(device="cpu")
        texts = [f"text{i}" for i in range(10)]
        result = embedder.encode(texts, batch_size=3)

        assert result.shape == (10, 384)
        assert tokenizer.call_count == 4  # 10 texts / 3 batch_size = 4 batches

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_encode_with_normalization(self, mock_tokenizer_class, mock_model_class):
        """Test de l'encodage avec normalisation."""
        tokenizer = Mock()

        def tokenizer_call(*args, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (1, 10)),
                "attention_mask": torch.ones((1, 10), dtype=torch.long),
            }

        tokenizer.side_effect = tokenizer_call
        mock_tokenizer_class.from_pretrained.return_value = tokenizer

        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model

        def model_call(**kwargs):
            output = Mock()
            output.last_hidden_state = torch.randn(1, 10, 384)
            return output

        model.side_effect = model_call
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding(device="cpu")
        result = embedder.encode("Hello", normalize=True)

        # Vérifier que le vecteur est normalisé
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_encode_error_handling(self, mock_tokenizer_class, mock_model_class):
        """Test de la gestion des erreurs pendant l'encodage."""
        tokenizer = Mock()
        tokenizer.side_effect = Exception("Tokenization error")
        mock_tokenizer_class.from_pretrained.return_value = Mock()

        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding()
        embedder.tokenizer = tokenizer

        with pytest.raises(RuntimeError, match="Erreur lors de l'encoding"):
            embedder.encode("text")

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_embedding_dim_property(self, mock_tokenizer_class, mock_model_class):
        """Test de la propriété embedding_dim."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        model = Mock()
        model.config.hidden_size = 768
        model.to.return_value = model
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding()
        assert embedder.embedding_dim == 768

    @patch("stratadl.core.embedding.huggingface.AutoModel")
    @patch("stratadl.core.embedding.huggingface.AutoTokenizer")
    def test_repr(self, mock_tokenizer_class, mock_model_class):
        """Test de la méthode __repr__."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        model = Mock()
        model.config.hidden_size = 384
        model.to.return_value = model
        mock_model_class.from_pretrained.return_value = model

        embedder = HuggingFaceEmbedding(model_name="test-model")
        repr_str = repr(embedder)

        assert "HuggingFaceEmbedding" in repr_str
        assert "test-model" in repr_str
        assert "384" in repr_str
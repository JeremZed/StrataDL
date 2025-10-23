from typing import Union, List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from stratadl.core.embedding.base import BaseEmbeddingModel


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Modèle d'embedding utilisant sentence-transformers.
    Plus optimisé que HuggingFace brut pour les embeddings de phrases.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers n'est pas installé. "
                "Installez-le avec: pip install sentence-transformers"
            )

        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors du chargement du modèle {model_name}: {str(e)}"
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """Encode des textes en embeddings."""
        texts = self._validate_texts(texts)

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'encoding: {str(e)}")

    @property
    def embedding_dim(self) -> int:
        """Retourne la dimension des embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedding(model={self.model_name}, dim={self.embedding_dim})"

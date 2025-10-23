from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import numpy.typing as npt


class BaseEmbeddingModel(ABC):
    """
    Interface abstraite pour tous les modèles d'embedding.
    """

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = False,
    ) -> npt.NDArray[np.float32]:
        """
        Encode un ou plusieurs textes en embeddings.

        Args:
            texts: Texte unique ou liste de textes à encoder
            batch_size: Taille des batchs pour le traitement
            show_progress: Afficher une barre de progression
            normalize: Normaliser les embeddings (L2)

        Returns:
            Array numpy de shape (n_texts, embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Retourne la dimension des embeddings produits."""
        pass

    def _validate_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """Valide et normalise l'input."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("La liste de textes ne peut pas être vide")
        return texts
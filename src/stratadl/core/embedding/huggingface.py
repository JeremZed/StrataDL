from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from stratadl.core.embedding.base import BaseEmbeddingModel


class HuggingFaceEmbedding(BaseEmbeddingModel):
    """
    Modèle d'embedding utilisant HuggingFace Transformers.
    Supporte le mean pooling avec attention mask et la normalisation pour aggréger les vecteurs en un seul par phrase.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: Nom du modèle HuggingFace
            device: Device à utiliser ('cuda', 'cpu', ou None pour auto-détection)
            max_length: Longueur maximale des séquences
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Mode évaluation
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors du chargement du modèle {model_name}: {str(e)}"
            )

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling avec prise en compte de l'attention mask.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Encode des textes en embeddings avec gestion de batchs.
        """
        texts = self._validate_texts(texts)

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        try:
            for i in iterator:
                batch = texts[i : i + batch_size]

                # Tokenization
                encoded_input = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                # Déplacer sur le bon device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

                # Forward pass
                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                # Mean pooling
                embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )

                # Normalisation L2 si demandée
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

            # Concaténer tous les batchs
            final_embeddings = np.vstack(all_embeddings)
            return final_embeddings.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'encoding: {str(e)}")

    @property
    def embedding_dim(self) -> int:
        """Retourne la dimension des embeddings."""
        return self.model.config.hidden_size

    def __repr__(self) -> str:
        return f"HuggingFaceEmbedding(model={self.model_name}, device={self.device}, dim={self.embedding_dim})"

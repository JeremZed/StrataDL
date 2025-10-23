from typing import Literal, Optional, Union
from stratadl.core.embedding.base import BaseEmbeddingModel
from stratadl.core.embedding.huggingface import HuggingFaceEmbedding
from stratadl.core.embedding.sentence_transformer import SentenceTransformerEmbedding


def get_embedding_model(
    backend: Literal["huggingface", "sentence_transformer"] = "sentence_transformer",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> BaseEmbeddingModel:
    """
    Factory pour créer un modèle d'embedding.

    Args:
        backend: Type de backend ('huggingface', 'sentence_transformer', 'custom')
        model_name: Nom du modèle à charger (optionnel, utilise des défauts)
        device: Device à utiliser ('cuda', 'cpu', ou None pour auto)
        **kwargs: Arguments additionnels spécifiques au backend

    Returns:
        Instance de BaseEmbeddingModel

    Examples:
        # Méthode standard avec sentence_transformer
        embedder = get_embedding_model("sentence_transformer")

        # En utilisant HuggingFace avec modèle spécifique
        embedder = get_embedding_model(
                "huggingface",
                model_name="bert-base-uncased",
                device="cuda"
            )

        # Ou alors avec un Custom model
        embedder = get_embedding_model(
                "custom",
                vocab_size=50000,
                embedding_dim=512,
                model_path="path/to/weights.pt"
            )
    """
    if backend == "huggingface":
        default_model = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(
            model_name=model_name or default_model, device=device, **kwargs
        )

    elif backend == "sentence_transformer":
        default_model = "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedding(
            model_name=model_name or default_model, device=device, **kwargs
        )

    else:
        raise ValueError(
            f"Backend '{backend}' inconnu. "
            f"Backends disponibles: huggingface, sentence_transformer"
        )


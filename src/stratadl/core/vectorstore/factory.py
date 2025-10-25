from typing import Literal
from .base import BaseVectorStore
from stratadl.core.vectorstore.chroma import ChromaDBStore

def get_vectorstore(
    backend: Literal["chromadb"] = "chromadb",
    **kwargs
) -> BaseVectorStore:
    """
    Retourne une instance du vector store choisi.
    Exemple :
        store = get_vectorstore("chromadb", collection_name="docs")
    """
    if backend == "chromadb":
        return ChromaDBStore(**kwargs)
    else:
        raise ValueError(f"Backend vectoriel inconnu : {backend}")


"""
Exemple d'utilisation du factory pour obtenir un vector store.
# src/core/vectorstore/example_usage.py

from core.vectorstore.factory import get_vectorstore

store = get_vectorstore("chromadb", collection_name="test")

store.add_documents([
    {
        "id": "1",
        "content": "Les réseaux de neurones convolutifs sont efficaces pour la vision.",
        "embedding": [0.1, 0.2, 0.3],  # exemple
        "metadata": {"source": "article1"},
    }
])

print("Count:", store.count())

query_vec = [0.1, 0.25, 0.35]
results = store.query(query_vec, top_k=3)
print("Results:", results)

"""
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

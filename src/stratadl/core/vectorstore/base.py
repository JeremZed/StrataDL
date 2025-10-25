"""
Interface definitions for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

from dataclasses import dataclass


@dataclass
class QueryResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] | None = None


class BaseVectorStore(ABC):
    """
    Interface pour les implémentations de bases vectorielles.
    """

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ajoute une liste de documents à la base vectorielle.

        Args:
            documents (List[Dict[str, Any]]): Liste de documents à ajouter.
        """
        pass

    @abstractmethod
    def query(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Interroge la base vectorielle avec un vecteur de requête.

        Args:
            query_vector (List[float]): Vecteur de requête.
            top_k (int): Nombre de résultats à retourner.

        Returns:
            List[Dict[str, Any]]: Documents avec leurs métadonnées et scores de similarité
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Supprime des documents de la base vectorielle par leurs identifiants.

        Args:
            ids (List[str]): Liste des identifiants des documents à supprimer.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Retourne le nombre total de documents dans la base vectorielle.

        Returns:
            int: Nombre total de documents.
        """
        pass

    @abstractmethod
    def update(self, document_id: str, document: Dict[str, Any]) -> None:
        """
        Met à jour un document existant dans la base vectorielle.
        Args:
            document_id (str): Identifiant du document à mettre à jour.
            document (Dict[str, Any]): Nouveau contenu du document.
        Returns:
            None
        """
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Dict[str, Any] | None:
        """
        Récupère un document par son identifiant.
        Args:
            doc_id (str): Identifiant du document à récupérer.
        Returns:
            Dict[str, Any] | None: Document récupéré ou None s'il n'existe
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Supprime tous les documents de la base vectorielle.
        Returns:
            None
        """
        pass
from typing import List, Dict, Any, Optional
from stratadl.core.vectorstore.base import BaseVectorStore, QueryResult
from chromadb import PersistentClient

import logging

logger = logging.getLogger(__name__)

class ChromaDBStore(BaseVectorStore):
    """
    Implémentation du vector store utilisant ChromaDB.
    """

    def __init__(self, collection_name: str = "default_collection", persist_directory: str = ".chromadb"):
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection ChromaDB '{collection_name}' initialisée")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ajoute une liste de documents à la base vectorielle ChromaDB.
        """

        try:
            if not documents:
                raise ValueError("La liste de documents est vide")

            for doc in documents:
                if "id" not in doc or "embedding" not in doc:
                    raise ValueError(f"Document invalide : {doc}")


            ids = [str(d["id"]) for d in documents]
            embeddings = [d["embedding"] for d in documents]
            metadatas = [d.get("metadata", None) for d in documents]
            contents = [d.get("content", "") for d in documents]

            self.collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
            logger.info(f"{len(documents)} document(s) ajouté(s) à la collection")

        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'ajout des documents : {e}")

    def delete(self, ids: List[str]) -> None:
        """
        Supprime des documents de la base vectorielle ChromaDB par leurs identifiants.
        """
        if not ids:
            logger.warning("Tentative de suppression avec une liste vide d'IDs")
            return

        self.collection.delete(ids=ids)
        logger.info(f"{len(ids)} document(s) supprimé(s)")

    def count(self) -> int:
        """
        Compte le nombre de documents dans la base vectorielle ChromaDB.
        """
        return self.collection.count()

    def update(self, document_id: str, document: Dict[str, Any]) -> None:
        """
        Met à jour un document existant dans la base vectorielle.

        Args:
            document_id (str): Identifiant du document à mettre à jour.
            document (Dict[str, Any]): Nouveau contenu du document.
                Doit contenir 'embedding' et peut contenir 'content' et 'metadata'.

        Raises:
            ValueError: Si le document ne contient pas les champs requis.
            RuntimeError: Si la mise à jour échoue.
        """
        try:

            # Checkup
            existing = self.get_by_id(document_id)
            if existing is None:
                raise ValueError(f"Document avec l'ID '{document_id}' introuvable")

            if "embedding" not in document:
                raise ValueError("Le document doit contenir un champ 'embedding'")

            update_data = {
                "ids": [str(document_id)],
                "embeddings": [document["embedding"]]
            }

            if "content" in document:
                update_data["documents"] = [document["content"]]

            if "metadata" in document:
                update_data["metadatas"] = [document["metadata"]]

            self.collection.update(**update_data)
            logger.info(f"Document '{document_id}' mis à jour avec succès")

        except ValueError as e:
            logger.error(f"Erreur de validation lors de la mise à jour : {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du document '{document_id}' : {e}")
            raise RuntimeError(f"Échec de la mise à jour : {e}")

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un document par son identifiant.

        Args:
            doc_id (str): Identifiant du document à récupérer.

        Returns:
            Dict[str, Any] | None: Document récupéré avec ses champs (id, content, metadata, embedding)
                                   ou None si le document n'existe pas.
        """
        try:
            result = self.collection.get(
                ids=[str(doc_id)],
                include=["embeddings", "documents", "metadatas"]
            )

            if not result["ids"]:
                logger.debug(f"Document '{doc_id}' introuvable")
                return None

            document = QueryResult(
                id=result["ids"][0],
                content=result["documents"][0],
                metadata=result["metadatas"][0],
                score=None
            )

            logger.debug(f"Document '{doc_id}' récupéré avec succès")
            return document

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document '{doc_id}' : {e}")
            return None

    def clear(self) -> None:
        """
        Supprime tous les documents de la base vectorielle.

        Note: Cette opération est irréversible.
        """
        try:

            all_ids = self.collection.get()["ids"]

            if not all_ids:
                logger.info("La collection est déjà vide")
                return

            self.collection.delete(ids=all_ids)
            logger.info(f"Collection vidée : {len(all_ids)} document(s) supprimé(s)")

        except Exception as e:
            logger.error(f"Erreur lors du vidage de la collection : {e}")
            raise RuntimeError(f"Échec du vidage de la collection : {e}")

    def query(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        sort_by_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recherche avancée de documents selon leurs métadonnées, avec ou sans vecteur de requête.

        Returns:
            List[QueryResult]: Liste d'objets contenant id, content, metadata et score.
        """
        try:

            logging.debug(f"Recherche avancée - Filtres: {filters}, vector={query_vector}")

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filters
            )

            docs: List[QueryResult] = []

            for i in range(len(results["ids"][0])):
                docs.append(QueryResult(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=results["distances"][0][i]
                ))

            # trie des information par score
            if sort_by_score and query_vector is not None:
                docs.sort(key=lambda d: d.score if d.score is not None else float("inf"))

            return docs

        except Exception as e:
            logger.error(f"Erreur lors de la recherche par métadonnées : {e}", exc_info=True)
            raise RuntimeError(f"Échec de la recherche par métadonnées : {e}")

"""
tests/test_vectorstore.py
Tests unitaires pour les implémentations de vector store.
"""

import pytest
import tempfile
import shutil

from stratadl.core.vectorstore.base import BaseVectorStore, QueryResult
from stratadl.core.vectorstore.chroma import ChromaDBStore
from stratadl.core.vectorstore.factory import get_vectorstore


# Fixtures
@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def chroma_store(temp_dir):
    """Crée une instance de ChromaDBStore pour les tests."""
    store = ChromaDBStore(
        collection_name="test_collection",
        persist_directory=temp_dir
    )
    yield store
    # Nettoyage après chaque test
    try:
        store.clear()
    except:
        pass


@pytest.fixture
def sample_documents():
    """Fournit des documents de test."""
    return [
        {
            "id": "doc1",
            "content": "Les réseaux de neurones convolutifs sont efficaces pour la vision.",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {"source": "article1", "type": "research"}
        },
        {
            "id": "doc2",
            "content": "Le traitement du langage naturel utilise des transformers.",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            "metadata": {"source": "article2", "type": "research"}
        },
        {
            "id": "doc3",
            "content": "L'apprentissage par renforcement apprend des actions optimales.",
            "embedding": [0.3, 0.4, 0.5, 0.6, 0.7],
            "metadata": {"source": "article3", "type": "tutorial"}
        }
    ]


# Tests pour ChromaDBStore - Initialisation
class TestChromaDBStoreInit:
    """Tests d'initialisation de ChromaDBStore."""

    def test_init_default_parameters(self, temp_dir):
        """Test l'initialisation avec les paramètres par défaut."""
        store = ChromaDBStore(persist_directory=temp_dir)
        assert store.collection is not None
        assert store.client is not None

    def test_init_custom_collection_name(self, temp_dir):
        """Test l'initialisation avec un nom de collection personnalisé."""
        collection_name = "custom_collection"
        store = ChromaDBStore(
            collection_name=collection_name,
            persist_directory=temp_dir
        )
        assert store.collection.name == collection_name

    def test_init_persists_data(self, temp_dir):
        """Test que les données persistent entre les instances."""
        store1 = ChromaDBStore(
            collection_name="persist_test",
            persist_directory=temp_dir
        )
        store1.add_documents([{
            "id": "persist1",
            "content": "test",
            "embedding": [0.1, 0.2, 0.3]
        }])

        # Nouvelle instance avec la même collection
        store2 = ChromaDBStore(
            collection_name="persist_test",
            persist_directory=temp_dir
        )
        assert store2.count() == 1


# Tests pour add_documents
class TestAddDocuments:
    """Tests de la méthode add_documents."""

    def test_add_single_document(self, chroma_store):
        """Test l'ajout d'un seul document."""
        doc = {
            "id": "single",
            "content": "Test document",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"key": "value"}
        }
        chroma_store.add_documents([doc])
        assert chroma_store.count() == 1

    def test_add_multiple_documents(self, chroma_store, sample_documents):
        """Test l'ajout de plusieurs documents."""
        chroma_store.add_documents(sample_documents)
        assert chroma_store.count() == len(sample_documents)

    def test_add_document_without_metadata(self, chroma_store):
        """Test l'ajout d'un document sans métadonnées."""
        doc = {
            "id": "no_meta",
            "content": "Document without metadata",
            "embedding": [0.1, 0.2, 0.3]
        }
        chroma_store.add_documents([doc])
        result = chroma_store.get_by_id("no_meta")
        assert result is not None
        assert result.metadata == None

    def test_add_empty_list_raises_error(self, chroma_store):
        """Test qu'ajouter une liste vide lève une erreur."""
        with pytest.raises(RuntimeError, match="La liste de documents est vide"):
            chroma_store.add_documents([])

    def test_add_document_without_id_raises_error(self, chroma_store):
        """Test qu'un document sans ID lève une erreur."""
        doc = {
            "content": "No ID",
            "embedding": [0.1, 0.2, 0.3]
        }
        with pytest.raises(RuntimeError, match="Document invalide"):
            chroma_store.add_documents([doc])

    def test_add_document_without_embedding_raises_error(self, chroma_store):
        """Test qu'un document sans embedding lève une erreur."""
        doc = {
            "id": "no_embedding",
            "content": "No embedding"
        }
        with pytest.raises(RuntimeError, match="Document invalide"):
            chroma_store.add_documents([doc])

    def test_add_document_with_empty_content(self, chroma_store):
        """Test l'ajout d'un document avec un contenu vide."""
        doc = {
            "id": "empty_content",
            "embedding": [0.1, 0.2, 0.3]
        }
        chroma_store.add_documents([doc])
        result = chroma_store.get_by_id("empty_content")
        assert result is not None
        assert result.content == ""


# Tests pour delete
class TestDelete:
    """Tests de la méthode delete."""

    def test_delete_existing_document(self, chroma_store, sample_documents):
        """Test la suppression d'un document existant."""
        chroma_store.add_documents(sample_documents)
        initial_count = chroma_store.count()

        chroma_store.delete(["doc1"])
        assert chroma_store.count() == initial_count - 1
        assert chroma_store.get_by_id("doc1") is None

    def test_delete_multiple_documents(self, chroma_store, sample_documents):
        """Test la suppression de plusieurs documents."""
        chroma_store.add_documents(sample_documents)

        chroma_store.delete(["doc1", "doc2"])
        assert chroma_store.count() == 1
        assert chroma_store.get_by_id("doc3") is not None

    def test_delete_nonexistent_document(self, chroma_store, sample_documents):
        """Test la suppression d'un document qui n'existe pas."""
        chroma_store.add_documents(sample_documents)
        initial_count = chroma_store.count()

        # Ne devrait pas lever d'erreur
        chroma_store.delete(["nonexistent"])
        assert chroma_store.count() == initial_count

    def test_delete_empty_list(self, chroma_store, sample_documents):
        """Test la suppression avec une liste vide."""
        chroma_store.add_documents(sample_documents)
        initial_count = chroma_store.count()

        chroma_store.delete([])
        assert chroma_store.count() == initial_count


# Tests pour count
class TestCount:
    """Tests de la méthode count."""

    def test_count_empty_collection(self, chroma_store):
        """Test le comptage sur une collection vide."""
        assert chroma_store.count() == 0

    def test_count_after_adding_documents(self, chroma_store, sample_documents):
        """Test le comptage après ajout de documents."""
        chroma_store.add_documents(sample_documents)
        assert chroma_store.count() == len(sample_documents)

    def test_count_after_deletion(self, chroma_store, sample_documents):
        """Test le comptage après suppression."""
        chroma_store.add_documents(sample_documents)
        chroma_store.delete(["doc1"])
        assert chroma_store.count() == len(sample_documents) - 1


# Tests pour update
class TestUpdate:
    """Tests de la méthode update."""

    def test_update_existing_document(self, chroma_store, sample_documents):
        """Test la mise à jour d'un document existant."""
        chroma_store.add_documents(sample_documents)

        updated_doc = {
            "embedding": [0.9, 0.8, 0.7, 0.6, 0.5],
            "content": "Contenu mis à jour",
            "metadata": {"source": "updated", "type": "modified"}
        }

        chroma_store.update("doc1", updated_doc)
        result = chroma_store.get_by_id("doc1")

        assert result is not None
        assert result.content == "Contenu mis à jour"
        assert result.metadata["source"] == "updated"

    def test_update_only_embedding(self, chroma_store, sample_documents):
        """Test la mise à jour uniquement de l'embedding."""
        chroma_store.add_documents(sample_documents)
        original = chroma_store.get_by_id("doc1")

        updated_doc = {
            "embedding": [0.9, 0.8, 0.7, 0.6, 0.5]
        }

        chroma_store.update("doc1", updated_doc)
        result = chroma_store.get_by_id("doc1")

        assert result is not None
        assert result.content == original.content
        assert result.metadata == original.metadata

    def test_update_nonexistent_document_raises_error(self, chroma_store):
        """Test qu'updater un document inexistant lève une erreur."""
        updated_doc = {
            "embedding": [0.9, 0.8, 0.7]
        }

        with pytest.raises(ValueError, match="introuvable"):
            chroma_store.update("nonexistent", updated_doc)

    def test_update_without_embedding_raises_error(self, chroma_store, sample_documents):
        """Test qu'updater sans embedding lève une erreur."""
        chroma_store.add_documents(sample_documents)

        updated_doc = {
            "content": "New content"
        }

        with pytest.raises(ValueError, match="doit contenir un champ 'embedding'"):
            chroma_store.update("doc1", updated_doc)


# Tests pour get_by_id
class TestGetById:
    """Tests de la méthode get_by_id."""

    def test_get_existing_document(self, chroma_store, sample_documents):
        """Test la récupération d'un document existant."""
        chroma_store.add_documents(sample_documents)
        result = chroma_store.get_by_id("doc1")

        assert result is not None
        assert isinstance(result, QueryResult)
        assert result.id == "doc1"
        assert result.content == sample_documents[0]["content"]
        assert result.metadata == sample_documents[0]["metadata"]

    def test_get_nonexistent_document(self, chroma_store):
        """Test la récupération d'un document qui n'existe pas."""
        result = chroma_store.get_by_id("nonexistent")
        assert result is None

    def test_get_returns_query_result(self, chroma_store, sample_documents):
        """Test que get_by_id retourne un QueryResult."""
        chroma_store.add_documents(sample_documents)
        result = chroma_store.get_by_id("doc1")

        assert isinstance(result, QueryResult)
        assert hasattr(result, 'id')
        assert hasattr(result, 'content')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'score')
        assert result.score is None  # get_by_id ne calcule pas de score


# Tests pour clear
class TestClear:
    """Tests de la méthode clear."""

    def test_clear_non_empty_collection(self, chroma_store, sample_documents):
        """Test le vidage d'une collection non vide."""
        chroma_store.add_documents(sample_documents)
        assert chroma_store.count() > 0

        chroma_store.clear()
        assert chroma_store.count() == 0

    def test_clear_empty_collection(self, chroma_store):
        """Test le vidage d'une collection déjà vide."""
        assert chroma_store.count() == 0
        chroma_store.clear()  # Ne devrait pas lever d'erreur
        assert chroma_store.count() == 0

    def test_clear_is_irreversible(self, chroma_store, sample_documents):
        """Test que clear supprime définitivement les documents."""
        chroma_store.add_documents(sample_documents)
        chroma_store.clear()

        for doc in sample_documents:
            assert chroma_store.get_by_id(doc["id"]) is None


# Tests pour query
class TestQuery:
    """Tests de la méthode query."""

    def test_query_basic(self, chroma_store, sample_documents):
        """Test une requête basique."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert all(r.score is not None for r in results)

    def test_query_with_filters(self, chroma_store, sample_documents):
        """Test une requête avec filtres de métadonnées."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        filters = {"type": "research"}
        results = chroma_store.query(query_vector, filters=filters, top_k=5)

        assert len(results) <= 2  # Seulement 2 docs avec type="research"
        assert all(r.metadata["type"] == "research" for r in results)

    def test_query_top_k_limit(self, chroma_store, sample_documents):
        """Test que top_k limite bien les résultats."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=1)

        assert len(results) == 1

    def test_query_sorted_by_score(self, chroma_store, sample_documents):
        """Test que les résultats sont triés par score."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=3, sort_by_score=True)

        # Les scores doivent être en ordre croissant (distances)
        scores = [r.score for r in results]
        assert scores == sorted(scores)

    def test_query_returns_all_fields(self, chroma_store, sample_documents):
        """Test que query retourne tous les champs."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=1)

        result = results[0]
        assert result.id is not None
        assert result.content is not None
        assert result.metadata is not None
        assert result.score is not None

    def test_query_empty_collection(self, chroma_store):
        """Test une requête sur une collection vide."""
        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=5)

        assert len(results) == 0

    def test_query_with_no_matching_filters(self, chroma_store, sample_documents):
        """Test une requête avec des filtres qui ne matchent aucun document."""
        chroma_store.add_documents(sample_documents)

        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        filters = {"type": "nonexistent"}
        results = chroma_store.query(query_vector, filters=filters, top_k=5)

        assert len(results) == 0


# Tests pour factory
class TestFactory:
    """Tests pour la fonction factory get_vectorstore."""

    def test_get_chromadb_store(self, temp_dir):
        """Test la création d'un ChromaDBStore via factory."""
        store = get_vectorstore(
            "chromadb",
            collection_name="factory_test",
            persist_directory=temp_dir
        )

        assert isinstance(store, ChromaDBStore)
        assert isinstance(store, BaseVectorStore)

    def test_factory_invalid_backend_raises_error(self):
        """Test qu'un backend invalide lève une erreur."""
        with pytest.raises(ValueError, match="Backend vectoriel inconnu"):
            get_vectorstore("invalid_backend")

    def test_factory_default_backend(self, temp_dir):
        """Test le backend par défaut."""
        store = get_vectorstore(persist_directory=temp_dir)
        assert isinstance(store, ChromaDBStore)


# Tests d'intégration
class TestIntegration:
    """Tests d'intégration complets."""

    def test_full_workflow(self, chroma_store, sample_documents):
        """Test un workflow complet: add, query, update, delete."""
        # Ajout
        chroma_store.add_documents(sample_documents)
        assert chroma_store.count() == 3

        # Requête
        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        results = chroma_store.query(query_vector, top_k=2)
        assert len(results) == 2

        # Mise à jour
        updated_doc = {
            "embedding": [0.9, 0.8, 0.7, 0.6, 0.5],
            "content": "Updated content"
        }
        chroma_store.update("doc1", updated_doc)
        updated = chroma_store.get_by_id("doc1")
        assert updated.content == "Updated content"

        # Suppression
        chroma_store.delete(["doc2"])
        assert chroma_store.count() == 2

        # Clear
        chroma_store.clear()
        assert chroma_store.count() == 0

    def test_multiple_collections(self, temp_dir):
        """Test l'utilisation de plusieurs collections simultanément."""
        store1 = ChromaDBStore("collection1", temp_dir)
        store2 = ChromaDBStore("collection2", temp_dir)

        store1.add_documents([{
            "id": "c1_doc1",
            "content": "Collection 1",
            "embedding": [0.1, 0.2, 0.3]
        }])

        store2.add_documents([{
            "id": "c2_doc1",
            "content": "Collection 2",
            "embedding": [0.4, 0.5, 0.6]
        }])

        assert store1.count() == 1
        assert store2.count() == 1
        assert store1.get_by_id("c1_doc1") is not None
        assert store2.get_by_id("c2_doc1") is not None
        assert store1.get_by_id("c2_doc1") is None


# Tests de QueryResult
class TestQueryResult:
    """Tests pour la classe QueryResult."""

    def test_query_result_creation(self):
        """Test la création d'un QueryResult."""
        result = QueryResult(
            id="test_id",
            content="test content",
            metadata={"key": "value"},
            score=0.85
        )

        assert result.id == "test_id"
        assert result.content == "test content"
        assert result.metadata == {"key": "value"}
        assert result.score == 0.85

    def test_query_result_default_score(self):
        """Test que le score par défaut est None."""
        result = QueryResult(
            id="test_id",
            content="test content",
            metadata={}
        )

        assert result.score is None

    def test_query_result_is_dataclass(self):
        """Test que QueryResult est bien une dataclass."""
        result1 = QueryResult(
            id="1",
            content="content",
            metadata={},
            score=0.5
        )
        result2 = QueryResult(
            id="1",
            content="content",
            metadata={},
            score=0.5
        )

        assert result1 == result2


# Tests de performance (optionnels)
class TestPerformance:
    """Tests de performance basiques."""

    def test_large_batch_insertion(self, chroma_store):
        """Test l'insertion d'un grand nombre de documents."""
        large_batch = [
            {
                "id": f"doc_{i}",
                "content": f"Document {i}",
                "embedding": [float(i) * 0.1] * 5,
                "metadata": {"index": i}
            }
            for i in range(100)
        ]

        chroma_store.add_documents(large_batch)
        assert chroma_store.count() == 100

    def test_query_performance_with_many_documents(self, chroma_store):
        """Test les performances de requête avec beaucoup de documents."""
        large_batch = [
            {
                "id": f"doc_{i}",
                "content": f"Document {i}",
                "embedding": [float(i) * 0.01] * 5,
                "metadata": {"index": i}
            }
            for i in range(100)
        ]

        chroma_store.add_documents(large_batch)

        query_vector = [0.5, 0.5, 0.5, 0.5, 0.5]
        results = chroma_store.query(query_vector, top_k=10)

        assert len(results) == 10
        assert all(isinstance(r, QueryResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
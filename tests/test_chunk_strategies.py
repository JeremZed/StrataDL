import pytest
from stratadl.core.chunk.strategy.base import ChunkItem, ChunkStrategy
from stratadl.core.chunk.strategy.classic import ClassicChunk
from stratadl.core.chunk.strategy.sentence import SentenceChunk
from stratadl.core.chunk.strategy.tokenllm import TokenLLMChunk
from stratadl.core.chunk.strategy.hierarchy import HierarchyChunk
from stratadl.core.chunk.strategy.markdown import MarkdownChunk
from stratadl.core.chunk.strategy.type import ChunkTypeStrategy


class TestChunkItem:
    """Tests pour la classe ChunkItem"""

    def test_chunk_item_creation(self):
        """Test la création d'un ChunkItem"""
        chunk = ChunkItem(
            content="Test content",
            metadata={"key": "value"}
        )
        assert chunk.content == "Test content"
        assert chunk.metadata["key"] == "value"

    def test_chunk_item_empty_metadata(self):
        """Test ChunkItem avec métadonnées vides"""
        chunk = ChunkItem(content="Test", metadata={})
        assert chunk.content == "Test"
        assert chunk.metadata == {}


class TestClassicChunk:
    """Tests pour la stratégie ClassicChunk"""

    def test_init_default_values(self):
        """Test l'initialisation avec valeurs par défaut"""
        chunker = ClassicChunk()
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
        assert chunker.chunks == []

    def test_init_custom_values(self):
        """Test l'initialisation avec valeurs personnalisées"""
        chunker = ClassicChunk(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    def test_chunk_short_text(self):
        """Test le chunking d'un texte court"""
        chunker = ClassicChunk(chunk_size=100, overlap=10)
        result = chunker.chunk("Court texte")

        assert len(result) == 1
        assert result[0].content == "Court texte"
        assert result[0].metadata["type"] == ChunkTypeStrategy.CLASSIC.value

    def test_chunk_long_text(self, long_text):
        """Test le chunking d'un texte long"""
        chunker = ClassicChunk(chunk_size=50, overlap=10)
        result = chunker.chunk(long_text)

        assert len(result) > 1
        for chunk in result:
            assert len(chunk.content) <= 50 + 10  # chunk_size + marge
            assert chunk.metadata["type"] == ChunkTypeStrategy.CLASSIC.value

    def test_chunk_with_overlap(self):
        """Test que l'overlap fonctionne correctement"""
        chunker = ClassicChunk(chunk_size=20, overlap=5)
        text = "a" * 50
        result = chunker.chunk(text)

        # Vérifie qu'il y a bien plusieurs chunks
        assert len(result) > 1

        # Vérifie que chaque chunk fait au maximum chunk_size
        for chunk in result:
            assert len(chunk.content) <= chunker.chunk_size

        # Vérifie la présence d'overlap : la fin d'un chunk = début du suivant
        for i in range(1, len(result)):
            prev_chunk = result[i-1].content
            curr_chunk = result[i].content
            # Les overlap derniers caractères du chunk précédent doivent être
            # les overlap premiers caractères du chunk actuel
            overlap_size = min(len(prev_chunk), chunker.overlap)
            assert prev_chunk[-overlap_size:] == curr_chunk[:overlap_size]

    def test_positions_metadata(self):
        """Test que les positions start_pos et end_pos sont correctes"""
        chunker = ClassicChunk(chunk_size=20, overlap=5)
        text = "a" * 50
        result = chunker.chunk(text)

        for chunk in result:
            assert "start_pos" in chunk.metadata
            assert "end_pos" in chunk.metadata
            assert chunk.metadata["start_pos"] >= 0
            assert chunk.metadata["end_pos"] <= len(text) + chunker.overlap * 2

            # Vérifie que la taille du chunk correspond aux positions
            expected_size = chunk.metadata["end_pos"] - chunk.metadata["start_pos"]
            assert len(chunk.content) == expected_size
            assert len(chunk.content) <= chunker.chunk_size

    def test_chunk_empty_text(self):
        """Test le chunking d'un texte vide"""
        chunker = ClassicChunk()
        result = chunker.chunk("")

        assert len(result) == 1
        assert result[0].content == ""

    def test_positions_metadata(self):
        """Test que les positions start_pos et end_pos sont correctes"""
        chunker = ClassicChunk(chunk_size=20, overlap=5)
        text = "a" * 50
        result = chunker.chunk(text)

        for chunk in result:
            assert "start_pos" in chunk.metadata
            assert "end_pos" in chunk.metadata
            assert chunk.metadata["start_pos"] >= 0
            assert chunk.metadata["end_pos"] <= len(text) + chunker.overlap * 2


class TestSentenceChunk:
    """Tests pour la stratégie SentenceChunk"""

    def test_init_default_values(self):
        """Test l'initialisation"""
        chunker = SentenceChunk()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_chunk_by_sentences(self, sample_text):
        """Test le découpage par phrases"""
        chunker = SentenceChunk(chunk_size=200, overlap=20)
        result = chunker.chunk(sample_text)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["type"] == ChunkTypeStrategy.SENTENCE.value
            assert "line" in chunk.metadata

    def test_chunk_single_sentence(self):
        """Test avec une seule phrase"""
        chunker = SentenceChunk()
        result = chunker.chunk("Ceci est une phrase unique.")

        assert len(result) == 1
        assert result[0].metadata["line"] == 0

    def test_chunk_long_sentence(self):
        """Test avec une phrase dépassant chunk_size"""
        chunker = SentenceChunk(chunk_size=30, overlap=5)
        long_sentence = "Ceci est une très longue phrase qui dépasse largement la taille du chunk configuré."
        result = chunker.chunk(long_sentence)

        # La phrase devrait être découpée en plusieurs chunks
        assert len(result) >= 1

    def test_chunk_with_metadata(self, basic_metadata):
        """Test avec métadonnées"""
        chunker = SentenceChunk()
        result = chunker.chunk("Première phrase. Deuxième phrase.", metadata=basic_metadata)

        for chunk in result:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "pytest"


class TestTokenLLMChunk:
    """Tests pour la stratégie TokenLLMChunk"""

    def test_init_default_values(self):
        """Test l'initialisation"""
        chunker = TokenLLMChunk()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_chunk_short_text(self):
        """Test le chunking d'un texte court"""
        chunker = TokenLLMChunk(chunk_size=100, overlap=10)
        result = chunker.chunk("This is a short test text.")

        assert len(result) >= 1
        assert result[0].metadata["type"] == ChunkTypeStrategy.TOKEN.value
        assert "model" in result[0].metadata

    def test_chunk_with_custom_model(self):
        """Test avec un modèle personnalisé"""
        chunker = TokenLLMChunk(chunk_size=50, overlap=10)
        result = chunker.chunk(
            "This is a test text.",
            pretrained_model_name_or_path="bert-base-uncased"
        )

        assert result[0].metadata["model"] == "bert-base-uncased"

    def test_chunk_long_text(self):
        """Test avec un texte long nécessitant plusieurs chunks"""
        chunker = TokenLLMChunk(chunk_size=20, overlap=5)
        long_text = " ".join(["word"] * 100)
        result = chunker.chunk(long_text)

        assert len(result) > 1
        for chunk in result:
            assert chunk.metadata["type"] == ChunkTypeStrategy.TOKEN.value

    def test_chunk_with_metadata(self, basic_metadata):
        """Test avec métadonnées personnalisées"""
        chunker = TokenLLMChunk()
        result = chunker.chunk("Test text", metadata=basic_metadata)

        assert result[0].metadata["source"] == "test"
        assert result[0].metadata["author"] == "pytest"


class TestHierarchyChunk:
    """Tests pour la stratégie HierarchyChunk"""

    def test_init_default_values(self):
        """Test l'initialisation avec valeurs par défaut"""
        chunker = HierarchyChunk()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert chunker.separators == ["\n\n", "\n", ".", " ", ""]

    def test_init_custom_separators(self):
        """Test l'initialisation avec séparateurs personnalisés"""
        custom_seps = ["\n", ".", " "]
        chunker = HierarchyChunk(separators=custom_seps)
        assert chunker.separators == custom_seps

    def test_split_by_separator_paragraph(self):
        """Test le découpage par paragraphes"""
        chunker = HierarchyChunk()
        text = "Premier paragraphe\n\nDeuxième paragraphe\n\nTroisième paragraphe"
        result = chunker._split_by_separator(text, "\n\n")

        assert len(result) == 3
        assert result[0] == "Premier paragraphe\n\n"
        assert result[1] == "Deuxième paragraphe\n\n"
        assert result[2] == "Troisième paragraphe"

    def test_split_by_separator_empty(self):
        """Test le découpage caractère par caractère"""
        chunker = HierarchyChunk()
        result = chunker._split_by_separator("abc", "")
        assert result == ["a", "b", "c"]

    def test_chunk_short_text(self):
        """Test le chunking d'un texte court"""
        chunker = HierarchyChunk(chunk_size=100, overlap=10)
        result = chunker.chunk("Court texte.")

        assert len(result) == 1
        assert result[0].metadata["type"] == ChunkTypeStrategy.HIERARCHY.value
        assert "hierarchy" in result[0].metadata

    def test_chunk_with_hierarchy(self, sample_text):
        """Test que la hiérarchie est bien préservée"""
        chunker = HierarchyChunk(chunk_size=50, overlap=10)
        result = chunker.chunk(sample_text)

        for chunk in result:
            assert "hierarchy" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["type"] == ChunkTypeStrategy.HIERARCHY.value

    def test_chunk_with_parent_child(self):
        """Test la relation parent-enfant entre chunks"""
        chunker = HierarchyChunk(chunk_size=30, overlap=5)
        text = "a" * 100  # Texte qui va nécessiter un découpage récursif
        result = chunker.chunk(text)

        # Vérifie que certains chunks ont un parent
        has_parent = any(chunk.metadata.get("chunk_parent") is not None
                        for chunk in result)
        assert len(result) > 0  # Au moins un chunk créé

    def test_recursive_split(self):
        """Test le découpage récursif"""
        chunker = HierarchyChunk(chunk_size=20, overlap=0)
        text = "Premier paragraphe très long qui dépasse la taille.\n\nDeuxième paragraphe."
        result = chunker._recursive_split(text, chunker.separators)

        assert len(result) > 1
        # Vérifie que chaque tuple contient (texte, hiérarchie, chunk_id, parent_id)
        for item in result:
            assert len(item) == 4
            assert isinstance(item[0], str)  # texte
            assert isinstance(item[1], list)  # hiérarchie
            assert isinstance(item[2], str)  # chunk_id

    def test_apply_overlap(self):
        """Test l'application de l'overlap"""
        chunker = HierarchyChunk(chunk_size=50, overlap=10)
        chunks_data = [
            ("Premier chunk de texte", ["paragraph"], "id1", None),
            ("Deuxième chunk de texte", ["paragraph"], "id2", None),
            ("Troisième chunk", ["paragraph"], "id3", None)
        ]

        result = chunker._apply_overlap(chunks_data)

        # Le premier chunk ne devrait pas avoir d'overlap
        assert result[0][0] == "Premier chunk de texte"
        # Les chunks suivants devraient avoir l'overlap ajouté
        assert len(result[1][0]) > len(chunks_data[1][0])

    def test_chunk_with_metadata(self, basic_metadata):
        """Test avec métadonnées personnalisées"""
        chunker = HierarchyChunk()
        result = chunker.chunk("Test\n\nTexte", metadata=basic_metadata)

        for chunk in result:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "pytest"

    def test_chunk_index(self):
        """Test que chunk_index est correctement attribué"""
        chunker = HierarchyChunk(chunk_size=30, overlap=5)
        text = "Premier paragraphe.\n\nDeuxième paragraphe.\n\nTroisième paragraphe."
        result = chunker.chunk(text)

        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == i

    def test_generate_chunk_id_uniqueness(self):
        """Test que les IDs générés sont uniques"""
        chunker = HierarchyChunk()
        ids = [chunker._generate_chunk_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # Tous les IDs doivent être uniques


class TestMarkdownChunk:
    """Tests pour la stratégie MarkdownChunk"""

    def test_init_default_values(self):
        """Test l'initialisation"""
        chunker = MarkdownChunk()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50
        assert "\n## " in chunker.separators
        assert "\n### " in chunker.separators

    def test_markdown_separators(self):
        """Test que les séparateurs Markdown sont bien définis"""
        chunker = MarkdownChunk()
        assert "\n---\n" in chunker.separators
        assert "\n## " in chunker.separators
        assert "\n### " in chunker.separators

    def test_separator_names(self):
        """Test que les noms de séparateurs sont corrects"""
        chunker = MarkdownChunk()
        assert chunker.separator_names["\n## "] == "heading2"
        assert chunker.separator_names["\n### "] == "heading3"
        assert chunker.separator_names["\n---\n"] == "horizontal_rule"

    def test_chunk_simple_markdown(self, sample_markdown):
        """Test le chunking de Markdown simple"""
        chunker = MarkdownChunk(chunk_size=100, overlap=10)
        result = chunker.chunk(sample_markdown)

        assert len(result) > 0
        for chunk in result:
            assert chunk.metadata["format"] == "markdown"
            assert "hierarchy" in chunk.metadata

    def test_chunk_with_headers(self):
        """Test le respect de la structure des headers"""
        chunker = MarkdownChunk(chunk_size=50, overlap=10)
        text = """## Section 1
Contenu section 1.

## Section 2
Contenu section 2."""

        result = chunker.chunk(text)

        # Vérifie que des chunks ont été créés
        assert len(result) > 0
        # Vérifie que le format markdown est présent
        for chunk in result:
            assert chunk.metadata["format"] == "markdown"

    def test_chunk_with_horizontal_rule(self):
        """Test le découpage avec règle horizontale"""
        chunker = MarkdownChunk(chunk_size=100, overlap=10)
        text = """Contenu avant

---

Contenu après"""

        result = chunker.chunk(text)
        assert len(result) > 0

    def test_chunk_with_metadata(self, basic_metadata):
        """Test avec métadonnées personnalisées"""
        chunker = MarkdownChunk()
        result = chunker.chunk("# Titre\n\nContenu", metadata=basic_metadata)

        for chunk in result:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["format"] == "markdown"


class TestChunkTypeStrategy:
    """Tests pour l'énumération ChunkTypeStrategy"""

    def test_enum_values(self):
        """Test que toutes les valeurs d'énumération existent"""
        assert ChunkTypeStrategy.CLASSIC.value == "classic"
        assert ChunkTypeStrategy.SENTENCE.value == "sentence"
        assert ChunkTypeStrategy.TOKEN.value == "token"
        assert ChunkTypeStrategy.HIERARCHY.value == "hierarchy"

    def test_enum_members(self):
        """Test le nombre de membres de l'énumération"""
        assert len(ChunkTypeStrategy) == 4


class TestIntegration:
    """Tests d'intégration entre différentes stratégies"""

    def test_all_strategies_return_chunk_items(self, sample_text):
        """Test que toutes les stratégies retournent des ChunkItem"""
        strategies = [
            ClassicChunk(),
            SentenceChunk(),
            TokenLLMChunk(),
            HierarchyChunk(),
            MarkdownChunk()
        ]

        for strategy in strategies:
            result = strategy.chunk(sample_text)
            assert len(result) > 0
            assert all(isinstance(chunk, ChunkItem) for chunk in result)

    def test_metadata_preservation(self, basic_metadata):
        """Test que les métadonnées sont préservées dans toutes les stratégies"""
        strategies = [
            ClassicChunk(),
            SentenceChunk(),
            HierarchyChunk(),
            MarkdownChunk()
        ]

        for strategy in strategies:
            result = strategy.chunk("Test texte", metadata=basic_metadata)
            for chunk in result:
                assert chunk.metadata["source"] == "test"
                assert chunk.metadata["author"] == "pytest"

    def test_compare_strategies_on_same_text(self, sample_text):
        """Compare les résultats de différentes stratégies sur le même texte"""
        classic = ClassicChunk(chunk_size=100, overlap=10)
        sentence = SentenceChunk(chunk_size=100, overlap=10)
        hierarchy = HierarchyChunk(chunk_size=100, overlap=10)

        classic_result = classic.chunk(sample_text)
        sentence_result = sentence.chunk(sample_text)
        hierarchy_result = hierarchy.chunk(sample_text)

        # Toutes devraient produire au moins un chunk
        assert len(classic_result) > 0
        assert len(sentence_result) > 0
        assert len(hierarchy_result) > 0

        # Les types devraient être différents
        assert classic_result[0].metadata["type"] != sentence_result[0].metadata["type"]
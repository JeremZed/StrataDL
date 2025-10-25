# tests/core/ingestion/file/test_word.py

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from stratadl.core.ingestion.file.word import WordDocumentIngestor


class TestWordDocumentIngestor:
    """Tests pour la classe WordDocumentIngestor"""

    def test_init(self):
        """Test l'initialisation de la classe"""
        ingestor = WordDocumentIngestor("/path/to/document.docx")

        assert ingestor.file_path == "/path/to/document.docx"
        assert ingestor.output_dir == "output"
        assert ingestor.html_version is None
        assert ingestor.markdown_version is None

    def test_init_with_custom_output_dir(self):
        """Test l'initialisation avec un répertoire personnalisé"""
        ingestor = WordDocumentIngestor("/path/to/document.docx", "custom_output")

        assert ingestor.file_path == "/path/to/document.docx"
        assert ingestor.output_dir == "custom_output"

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_html(self, mock_mammoth, mock_file):
        """Test la conversion d'un document Word en HTML"""
        # Configuration du mock
        mock_result = Mock()
        mock_result.value = "<p>Test content</p>"
        mock_mammoth.return_value = mock_result

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert_to_html()

        # Vérifie que le fichier a été ouvert en mode binaire
        mock_file.assert_called_once_with("/path/to/document.docx", "rb")

        # Vérifie que mammoth a été appelé
        mock_mammoth.assert_called_once()

        # Vérifie le résultat
        expected_html = "<html><body><p>Test content</p></body></html>"
        assert result == expected_html
        assert ingestor.html_version == expected_html

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_html_stores_result(self, mock_mammoth, mock_file):
        """Test que le résultat HTML est stocké dans l'instance"""
        mock_result = Mock()
        mock_result.value = "<h1>Title</h1><p>Content</p>"
        mock_mammoth.return_value = mock_result

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        ingestor.convert_to_html()

        assert ingestor.html_version is not None
        assert "<h1>Title</h1>" in ingestor.html_version
        assert "<html><body>" in ingestor.html_version

    @patch('stratadl.core.ingestion.file.word.md')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_markdown_with_existing_html(self, mock_mammoth, mock_file, mock_md):
        """Test la conversion HTML vers Markdown avec HTML existant"""
        mock_md.return_value = "# Title\n\nContent"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        ingestor.html_version = "<html><body><h1>Title</h1><p>Content</p></body></html>"

        result = ingestor.convert_to_markdown()

        # Vérifie que mammoth n'est pas appelé (HTML déjà existant)
        mock_mammoth.assert_not_called()

        # Vérifie que markdownify a été appelé avec les bons paramètres
        mock_md.assert_called_once_with(
            "<html><body><h1>Title</h1><p>Content</p></body></html>",
            strip=['img']
        )

        # Vérifie le résultat
        assert result == "# Title\n\nContent"
        assert ingestor.markdown_version == "# Title\n\nContent"

    @patch('stratadl.core.ingestion.file.word.md')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_markdown_without_html(self, mock_mammoth, mock_file, mock_md):
        """Test la conversion Markdown sans HTML préexistant"""
        mock_result = Mock()
        mock_result.value = "<h1>Title</h1><p>Content</p>"
        mock_mammoth.return_value = mock_result
        mock_md.return_value = "# Title\n\nContent"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert_to_markdown()

        # Vérifie que mammoth a été appelé (HTML n'existait pas)
        mock_mammoth.assert_called_once()

        # Vérifie que markdownify a été appelé
        mock_md.assert_called_once()

        assert result == "# Title\n\nContent"

    @patch('stratadl.core.ingestion.file.word.md')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_markdown_with_custom_strip(self, mock_mammoth, mock_file, mock_md):
        """Test la conversion Markdown avec paramètre strip personnalisé"""
        mock_result = Mock()
        mock_result.value = "<h1>Title</h1>"
        mock_mammoth.return_value = mock_result
        mock_md.return_value = "# Title"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert_to_markdown(strip=['img', 'a'])

        # Vérifie que markdownify a été appelé avec le strip personnalisé
        mock_md.assert_called_once_with(
            "<html><body><h1>Title</h1></body></html>",
            strip=['img', 'a']
        )

    @patch('stratadl.core.ingestion.file.word.md')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_markdown_without_strip(self, mock_mammoth, mock_file, mock_md):
        """Test la conversion Markdown sans stripping d'éléments"""
        mock_result = Mock()
        mock_result.value = "<h1>Title</h1>"
        mock_mammoth.return_value = mock_result
        mock_md.return_value = "# Title"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert_to_markdown(strip=[])

        # Vérifie que markdownify a été appelé avec strip vide
        mock_md.assert_called_once_with(
            "<html><body><h1>Title</h1></body></html>",
            strip=[]
        )

    @patch('stratadl.core.ingestion.file.word.md')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_calls_convert_to_markdown(self, mock_mammoth, mock_file, mock_md):
        """Test que convert() appelle convert_to_markdown()"""
        mock_result = Mock()
        mock_result.value = "<p>Content</p>"
        mock_mammoth.return_value = mock_result
        mock_md.return_value = "Content"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert()

        # Vérifie que le résultat est le même que convert_to_markdown()
        assert result == "Content"
        assert ingestor.markdown_version == "Content"

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake docx content')
    @patch('stratadl.core.ingestion.file.word.mammoth.convert_to_html')
    def test_convert_to_html_with_complex_content(self, mock_mammoth, mock_file):
        """Test la conversion avec du contenu HTML complexe"""
        mock_result = Mock()
        complex_html = """
        <h1>Title</h1>
        <p>Paragraph with <strong>bold</strong> and <em>italic</em></p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
        mock_result.value = complex_html
        mock_mammoth.return_value = mock_result

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        result = ingestor.convert_to_html()

        assert "<h1>Title</h1>" in result
        assert "<strong>bold</strong>" in result
        assert "<ul>" in result

    @patch('stratadl.core.ingestion.file.word.md')
    def test_convert_to_markdown_preserves_html(self, mock_md):
        """Test que convert_to_markdown préserve la version HTML"""
        mock_md.return_value = "# Title"

        ingestor = WordDocumentIngestor("/path/to/document.docx")
        original_html = "<html><body><h1>Title</h1></body></html>"
        ingestor.html_version = original_html

        ingestor.convert_to_markdown()

        # Vérifie que la version HTML n'a pas été modifiée
        assert ingestor.html_version == original_html

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_convert_to_html_file_not_found(self, mock_file):
        """Test la gestion d'un fichier introuvable"""
        ingestor = WordDocumentIngestor("/path/to/nonexistent.docx")

        with pytest.raises(FileNotFoundError):
            ingestor.convert_to_html()

    @patch('builtins.open', side_effect=PermissionError)
    def test_convert_to_html_permission_error(self, mock_file):
        """Test la gestion d'une erreur de permission"""
        ingestor = WordDocumentIngestor("/path/to/document.docx")

        with pytest.raises(PermissionError):
            ingestor.convert_to_html()
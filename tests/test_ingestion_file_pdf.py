import pytest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os
from stratadl.core.ingestion.file.pdf import PDFDocumentIngestor


class TestPDFDocumentIngestor:
    """Tests unitaires pour PDFDocumentIngestor"""

    @pytest.fixture
    def temp_pdf_path(self):
        """Crée un fichier PDF temporaire pour les tests"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def output_dir(self):
        """Crée un répertoire de sortie temporaire"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def ingestor(self, temp_pdf_path, output_dir):
        """Fixture pour créer une instance de PDFDocumentIngestor"""
        return PDFDocumentIngestor(temp_pdf_path, output_dir)

    def test_init_default_output_dir(self, temp_pdf_path):
        ingestor = PDFDocumentIngestor(temp_pdf_path)
        assert ingestor.file_path == temp_pdf_path
        assert ingestor.output_dir == "output"
        assert ingestor.markdown_version is None

    def test_init_custom_output_dir(self, temp_pdf_path, output_dir):
        ingestor = PDFDocumentIngestor(temp_pdf_path, output_dir)
        assert ingestor.file_path == temp_pdf_path
        assert ingestor.output_dir == output_dir
        assert ingestor.markdown_version is None

    @patch("pymupdf4llm.to_markdown")
    def test_convert_to_markdown_success(self, mock_to_markdown, ingestor, temp_pdf_path):
        expected = "# Test"
        mock_to_markdown.return_value = expected

        result = ingestor.convert_to_markdown()

        mock_to_markdown.assert_called_once_with(
            temp_pdf_path,
            page_chunks=False,
            write_images=False,
            show_progress=False
        )
        assert result == expected
        assert ingestor.markdown_version == expected

    @patch("pymupdf4llm.to_markdown")
    def test_convert_to_markdown_exception(self, mock_to_markdown, ingestor):
        mock_to_markdown.side_effect = Exception("Erreur de conversion")
        with pytest.raises(Exception, match="Erreur de conversion"):
            ingestor.convert_to_markdown()

    @patch("pymupdf4llm.to_markdown")
    def test_convert_returns_markdown(self, mock_to_markdown, ingestor):
        mock_to_markdown.return_value = "# Doc"
        result = ingestor.convert()
        assert result == "# Doc"
        assert ingestor.markdown_version == "# Doc"

    @patch("requests.put")
    def test_parser_tika_default_url(self, mock_put, ingestor, temp_pdf_path):
        mock_meta_resp = MagicMock()
        mock_meta_resp.json.return_value = {"Author": "Test"}
        mock_text_resp = MagicMock()
        mock_text_resp.text = "Document content"
        mock_text_resp.status_code = 200

        mock_put.side_effect = [mock_text_resp, mock_meta_resp]

        metadata, content, status = ingestor.parser_tika()

        assert metadata == {"Author": "Test"}
        assert content == "Document content"
        assert status == 200

        calls = mock_put.call_args_list
        assert calls[0].args[0] == "http://localhost:9998/tika"
        assert calls[1].args[0] == "http://localhost:9998/meta"

    @patch("requests.put")
    def test_parser_tika_custom_url(self, mock_put, ingestor):
        mock_meta_resp = MagicMock()
        mock_meta_resp.json.return_value = {"meta": "ok"}
        mock_text_resp = MagicMock()
        mock_text_resp.text = "Content"
        mock_text_resp.status_code = 200
        mock_put.side_effect = [mock_text_resp, mock_meta_resp]

        meta, content, status = ingestor.parser_tika(tika_url="http://custom:9999")
        assert meta == {"meta": "ok"}
        assert content == "Content"
        assert status == 200

        assert mock_put.call_args_list[0].args[0] == "http://custom:9999/tika"
        assert mock_put.call_args_list[1].args[0] == "http://custom:9999/meta"

    @patch("requests.put")
    def test_parser_tika_http_error(self, mock_put, ingestor):
        bad_response = MagicMock()
        bad_response.status_code = 500
        bad_response.text = "Error"
        bad_response.json.return_value = {}
        mock_put.side_effect = [bad_response, bad_response]

        meta, content, status = ingestor.parser_tika()
        assert status == 500
        assert content == "Error"
        assert meta == {}

    @patch("requests.put")
    def test_parser_tika_exception(self, mock_put, ingestor):
        mock_put.side_effect = Exception("Connexion refusée")
        with pytest.raises(Exception, match="Connexion refusée"):
            ingestor.parser_tika()

    def test_inherits_from_base_file_ingestor(self, ingestor):
        from stratadl.core.ingestion.file.base import BaseFileIngestor
        assert isinstance(ingestor, BaseFileIngestor)

# tests/core/ingestion/file/test_base.py

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from stratadl.core.ingestion.file.base import BaseFileIngestor


class ConcreteFileIngestor(BaseFileIngestor):
    """Implémentation concrète pour tester la classe abstraite"""

    def convert(self) -> str:
        return "# Test Markdown\n\nThis is a test."


class TestBaseFileIngestor:
    """Tests pour la classe BaseFileIngestor"""

    def test_init_with_default_output_dir(self):
        """Test l'initialisation avec le répertoire de sortie par défaut"""
        ingestor = ConcreteFileIngestor("/path/to/file.txt")

        assert ingestor.file_path == "/path/to/file.txt"
        assert ingestor.output_dir == "output"

    def test_init_with_custom_output_dir(self):
        """Test l'initialisation avec un répertoire de sortie personnalisé"""
        ingestor = ConcreteFileIngestor("/path/to/file.txt", "custom_output")

        assert ingestor.file_path == "/path/to/file.txt"
        assert ingestor.output_dir == "custom_output"

    def test_convert_is_abstract(self):
        """Test que convert() est une méthode abstraite"""
        with pytest.raises(TypeError):
            BaseFileIngestor("/path/to/file.txt")

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_with_default_filename(self, mock_abspath, mock_file, mock_makedirs):
        """Test la sauvegarde avec le nom de fichier par défaut"""
        mock_abspath.return_value = "/absolute/path/output/file.md"

        ingestor = ConcreteFileIngestor("/path/to/file.txt", "output")
        result = ingestor.save()

        # Vérifie que le répertoire a été créé
        mock_makedirs.assert_called_once_with("output", exist_ok=True)

        # Vérifie que le fichier a été ouvert avec le bon nom
        expected_path = os.path.join("output", "file.md")
        mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

        # Vérifie que le contenu a été écrit
        mock_file().write.assert_called_once_with("# Test Markdown\n\nThis is a test.")

        # Vérifie que le chemin absolu est retourné
        assert result == "/absolute/path/output/file.md"

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_with_custom_filename(self, mock_abspath, mock_file, mock_makedirs):
        """Test la sauvegarde avec un nom de fichier personnalisé"""
        mock_abspath.return_value = "/absolute/path/output/custom.md"

        ingestor = ConcreteFileIngestor("/path/to/file.txt", "output")
        result = ingestor.save(filename="custom.md")

        # Vérifie que le fichier a été ouvert avec le nom personnalisé
        expected_path = os.path.join("output", "custom.md")
        mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

        assert result == "/absolute/path/output/custom.md"

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_creates_output_directory(self, mock_abspath, mock_file, mock_makedirs):
        """Test que le répertoire de sortie est créé s'il n'existe pas"""
        mock_abspath.return_value = "/absolute/path/nested/output/file.md"

        ingestor = ConcreteFileIngestor("/path/to/file.txt", "nested/output")
        ingestor.save()

        mock_makedirs.assert_called_once_with("nested/output", exist_ok=True)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_handles_file_with_multiple_extensions(self, mock_abspath, mock_file, mock_makedirs):
        """Test la sauvegarde d'un fichier avec plusieurs extensions"""
        mock_abspath.return_value = "/absolute/path/output/archive.tar.md"

        ingestor = ConcreteFileIngestor("/path/to/archive.tar.gz", "output")
        result = ingestor.save()

        # Vérifie que seule la dernière extension est remplacée
        expected_path = os.path.join("output", "archive.tar.md")
        mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_with_path_without_extension(self, mock_abspath, mock_file, mock_makedirs):
        """Test la sauvegarde d'un fichier sans extension"""
        mock_abspath.return_value = "/absolute/path/output/file.md"

        ingestor = ConcreteFileIngestor("/path/to/file", "output")
        result = ingestor.save()

        expected_path = os.path.join("output", "file.md")
        mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    def test_save_writes_correct_content(self, mock_abspath, mock_file, mock_makedirs):
        """Test que le contenu markdown correct est écrit"""
        mock_abspath.return_value = "/absolute/path/output/file.md"

        ingestor = ConcreteFileIngestor("/path/to/file.txt", "output")
        ingestor.save()

        # Vérifie que convert() a été appelé et son résultat écrit
        mock_file().write.assert_called_once_with("# Test Markdown\n\nThis is a test.")
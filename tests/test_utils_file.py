# tests/test_utils_file.py

import pytest
import os
import tempfile
import shutil
from pathlib import Path

from stratadl.core.utils.file import (
    get_files,
    create_directory,
    rename_files_with_content_hash
)


class TestGetFiles:
    @pytest.fixture
    def temp_dir(self):
        """Crée un dossier temporaire pour les tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_get_files_single_extension(self, temp_dir):
        """Test avec une seule extension"""
        # Crée des fichiers de test
        Path(os.path.join(temp_dir, "file1.txt")).touch()
        Path(os.path.join(temp_dir, "file2.txt")).touch()
        Path(os.path.join(temp_dir, "file3.jpg")).touch()

        files = get_files(temp_dir, extensions={'.txt'})
        assert len(files) == 2
        assert all(f.endswith('.txt') for f in files)

    def test_get_files_multiple_extensions(self, temp_dir):
        """Test avec plusieurs extensions"""
        Path(os.path.join(temp_dir, "file1.txt")).touch()
        Path(os.path.join(temp_dir, "file2.jpg")).touch()
        Path(os.path.join(temp_dir, "file3.png")).touch()

        files = get_files(temp_dir, extensions={'.txt', '.jpg'})
        assert len(files) == 2

    def test_get_files_recursive(self, temp_dir):
        """Test en mode récursif"""
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        Path(os.path.join(temp_dir, "file1.txt")).touch()
        Path(os.path.join(subdir, "file2.txt")).touch()

        files = get_files(temp_dir, extensions={'.txt'}, recursive=True)
        assert len(files) == 2

    def test_get_files_non_recursive(self, temp_dir):
        """Test en mode non récursif"""
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        Path(os.path.join(temp_dir, "file1.txt")).touch()
        Path(os.path.join(subdir, "file2.txt")).touch()

        files = get_files(temp_dir, extensions={'.txt'}, recursive=False)
        assert len(files) == 1

    def test_get_files_case_insensitive(self, temp_dir):
        """Test que les extensions sont insensibles à la casse"""
        Path(os.path.join(temp_dir, "file1.TXT")).touch()
        Path(os.path.join(temp_dir, "file2.txt")).touch()

        files = get_files(temp_dir, extensions={'.txt'})
        assert len(files) == 2

    def test_get_files_empty_directory(self, temp_dir):
        """Test avec un dossier vide"""
        files = get_files(temp_dir, extensions={'.txt'})
        assert len(files) == 0

    def test_get_files_no_matching_extensions(self, temp_dir):
        """Test sans fichiers correspondant aux extensions"""
        Path(os.path.join(temp_dir, "file1.jpg")).touch()
        Path(os.path.join(temp_dir, "file2.png")).touch()

        files = get_files(temp_dir, extensions={'.txt'})
        assert len(files) == 0

    def test_get_files_ignores_directories(self, temp_dir):
        """Test que les dossiers sont ignorés"""
        os.makedirs(os.path.join(temp_dir, "folder.txt"))
        Path(os.path.join(temp_dir, "file.txt")).touch()

        files = get_files(temp_dir, extensions={'.txt'})
        assert len(files) == 1


class TestCreateDirectory:
    def test_create_directory_new(self):
        """Test création d'un nouveau dossier"""
        temp_dir = tempfile.mktemp()
        try:
            create_directory(temp_dir)
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
        finally:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def test_create_directory_existing(self):
        """Test avec un dossier existant (ne doit pas lever d'exception)"""
        temp_dir = tempfile.mkdtemp()
        try:
            create_directory(temp_dir)
            assert os.path.exists(temp_dir)
        finally:
            os.rmdir(temp_dir)

    def test_create_directory_nested(self):
        """Test création de dossiers imbriqués"""
        temp_dir = tempfile.mktemp()
        nested_dir = os.path.join(temp_dir, "sub1", "sub2", "sub3")
        try:
            create_directory(nested_dir)
            assert os.path.exists(nested_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestRenameFilesWithContentHash:
    @pytest.fixture
    def temp_dir(self):
        """Crée un dossier temporaire pour les tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def create_file_with_content(self, path, content):
        """Crée un fichier avec un contenu spécifique"""
        with open(path, 'wb') as f:
            f.write(content)

    def test_rename_files_basic(self, temp_dir):
        """Test basique de renommage"""
        file_path = os.path.join(temp_dir, "test.txt")
        self.create_file_with_content(file_path, b"Hello World")

        mapping = rename_files_with_content_hash(temp_dir, algo="sha1")

        assert len(mapping) == 1
        assert file_path in mapping
        assert not os.path.exists(file_path)
        assert os.path.exists(mapping[file_path])

    def test_rename_files_preserves_extension(self, temp_dir):
        """Test que l'extension est préservée"""
        file_path = os.path.join(temp_dir, "test.jpg")
        self.create_file_with_content(file_path, b"Image data")

        mapping = rename_files_with_content_hash(temp_dir)

        new_path = mapping[file_path]
        assert new_path.endswith('.jpg')

    def test_rename_files_duplicates(self, temp_dir):
        """Test avec des fichiers en double"""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        content = b"Same content"
        self.create_file_with_content(file1, content)
        self.create_file_with_content(file2, content)

        mapping = rename_files_with_content_hash(temp_dir)

        # Un fichier devrait être renommé, l'autre devrait être marqué comme doublon
        renamed_count = sum(1 for old, new in mapping.items()
                          if old != new and "non renommé" not in new)
        duplicates = sum(1 for v in mapping.values() if "non renommé" in v)

        assert renamed_count == 1
        assert duplicates == 1

    def test_rename_files_recursive(self, temp_dir):
        """Test en mode récursif"""
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)

        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")

        self.create_file_with_content(file1, b"Content 1")
        self.create_file_with_content(file2, b"Content 2")

        mapping = rename_files_with_content_hash(temp_dir)

        assert len(mapping) == 2

    def test_rename_files_different_algorithms(self, temp_dir):
        """Test avec différents algorithmes de hash"""
        file_path = os.path.join(temp_dir, "test.txt")
        self.create_file_with_content(file_path, b"Content")

        for algo in ['md5', 'sha1', 'sha256']:
            temp_subdir = os.path.join(temp_dir, algo)
            os.makedirs(temp_subdir)
            test_file = os.path.join(temp_subdir, "test.txt")
            self.create_file_with_content(test_file, b"Content")

            mapping = rename_files_with_content_hash(temp_subdir, algo=algo)
            assert len(mapping) == 1

    def test_rename_files_invalid_algorithm(self, temp_dir):
        """Test avec un algorithme invalide"""
        with pytest.raises(ValueError, match="Algorithme de hash inconnu"):
            rename_files_with_content_hash(temp_dir, algo="invalid_algo")

    def test_rename_files_already_named_correctly(self, temp_dir):
        """Test avec un fichier déjà nommé correctement"""
        content = b"Test content"
        import hashlib
        hash_val = hashlib.sha1(content).hexdigest()
        file_path = os.path.join(temp_dir, f"{hash_val}.txt")

        self.create_file_with_content(file_path, content)

        mapping = rename_files_with_content_hash(temp_dir)

        # Le fichier devrait être dans le mapping mais non renommé
        assert file_path in mapping
        assert mapping[file_path] == file_path

    def test_rename_files_empty_directory(self, temp_dir):
        """Test avec un dossier vide"""
        mapping = rename_files_with_content_hash(temp_dir)
        assert len(mapping) == 0

    def test_rename_files_binary_content(self, temp_dir):
        """Test avec du contenu binaire"""
        file_path = os.path.join(temp_dir, "binary.bin")
        binary_content = bytes(range(256))
        self.create_file_with_content(file_path, binary_content)

        mapping = rename_files_with_content_hash(temp_dir)

        assert len(mapping) == 1
        assert os.path.exists(mapping[file_path])
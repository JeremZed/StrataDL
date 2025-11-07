# tests/test_preprocessing_video.py

import pytest
import os
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from stratadl.core.preprocessing.video import split_to_frames, EXTENSIONS


class TestSplitToFrames:
    @pytest.fixture
    def temp_dirs(self):
        """Crée des dossiers temporaires pour les tests"""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        yield input_dir, output_dir
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

    def create_test_video(self, path, num_frames=10, size=(100, 100)):
        """Crée une vidéo de test"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 30.0, size)
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def test_split_to_frames_basic(self, temp_dirs):
        """Test basique d'extraction de frames"""
        input_dir, output_dir = temp_dirs
        video_path = os.path.join(input_dir, "test.mp4")
        self.create_test_video(video_path, num_frames=20)

        split_to_frames(input_dir, output_dir, frame_skip=5)

        # Vérifie que des fichiers ont été créés
        output_files = os.listdir(output_dir)
        assert len(output_files) > 0
        assert all(f.endswith('.jpg') for f in output_files)

    def test_split_to_frames_with_resize(self, temp_dirs):
        """Test avec redimensionnement"""
        input_dir, output_dir = temp_dirs
        video_path = os.path.join(input_dir, "test.mp4")
        self.create_test_video(video_path, num_frames=10)

        split_to_frames(input_dir, output_dir, frame_skip=2, resize_ratio=0.5)

        output_files = os.listdir(output_dir)
        assert len(output_files) > 0

    def test_split_to_frames_no_videos(self, temp_dirs):
        """Test quand aucune vidéo n'est trouvée"""
        input_dir, output_dir = temp_dirs

        with pytest.raises(Exception, match="Aucune vidéo trouvée"):
            split_to_frames(input_dir, output_dir)

    def test_split_to_frames_multiple_videos(self, temp_dirs):
        """Test avec plusieurs vidéos"""
        input_dir, output_dir = temp_dirs
        video1 = os.path.join(input_dir, "test1.mp4")
        video2 = os.path.join(input_dir, "test2.avi")

        self.create_test_video(video1, num_frames=10)
        self.create_test_video(video2, num_frames=10)

        split_to_frames(input_dir, output_dir, frame_skip=5)

        output_files = os.listdir(output_dir)
        assert len(output_files) > 0

    def test_split_to_frames_deduplication(self, temp_dirs):
        """Test que les frames identiques ne sont pas dupliquées"""
        input_dir, output_dir = temp_dirs
        video_path = os.path.join(input_dir, "test.mp4")

        # Crée une vidéo avec des frames identiques
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (100, 100))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(20):
            out.write(frame)
        out.release()

        split_to_frames(input_dir, output_dir, frame_skip=1)

        # Devrait n'avoir qu'une seule frame unique
        output_files = os.listdir(output_dir)
        assert len(output_files) == 1

    def test_split_to_frames_frame_skip(self, temp_dirs):
        """Test que frame_skip fonctionne correctement"""
        input_dir, output_dir = temp_dirs
        video_path = os.path.join(input_dir, "test.mp4")

        # Crée une vidéo avec des frames différentes
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (100, 100))
        for i in range(10):
            frame = np.full((100, 100, 3), i * 25, dtype=np.uint8)
            out.write(frame)
        out.release()

        split_to_frames(input_dir, output_dir, frame_skip=3)

        output_files = os.listdir(output_dir)
        # Avec 10 frames et frame_skip=3, on devrait avoir 4 frames (0, 3, 6, 9)
        assert len(output_files) == 4

    @patch('cv2.VideoCapture')
    def test_split_to_frames_corrupted_video(self, mock_capture, temp_dirs):
        """Test avec une vidéo corrompue"""
        input_dir, output_dir = temp_dirs
        video_path = os.path.join(input_dir, "test.mp4")

        # Crée un fichier vide
        Path(video_path).touch()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        # Ne devrait pas lever d'exception mais logger une erreur
        split_to_frames(input_dir, output_dir)


class TestConstants:
    def test_extensions_constant(self):
        """Vérifie que la constante EXTENSIONS contient les bonnes extensions"""
        assert '.mp4' in EXTENSIONS
        assert '.avi' in EXTENSIONS
        assert '.mov' in EXTENSIONS
        assert '.mkv' in EXTENSIONS
        assert '.webm' in EXTENSIONS
        assert len(EXTENSIONS) == 10
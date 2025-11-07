# tests/test_preprocessing_image.py

import pytest
import numpy as np
from PIL import Image

from stratadl.core.preprocessing.image import (
    hash_frame,
    get_resize_dim_img_pil,
    get_resize_dim_img_np,
    resize_image,
    EXTENSION_IMG
)


class TestHashFrame:
    def test_hash_frame_consistent(self):
        """Vérifie que le même frame produit le même hash"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hash1 = hash_frame(frame)
        hash2 = hash_frame(frame)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produit 64 caractères hex

    def test_hash_frame_different_frames(self):
        """Vérifie que des frames différents produisent des hash différents"""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        hash1 = hash_frame(frame1)
        hash2 = hash_frame(frame2)
        assert hash1 != hash2

    def test_hash_frame_small_change(self):
        """Vérifie qu'un petit changement modifie le hash"""
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[0, 0] = [2, 2, 2]
        hash1 = hash_frame(frame1)
        hash2 = hash_frame(frame2)
        assert hash1 != hash2


class TestGetResizeDimImgPil:
    def test_resize_dim_pil_upscale(self):
        """Test agrandissement d'une image PIL"""
        img = Image.new('RGB', (100, 200))
        new_w, new_h = get_resize_dim_img_pil(img, 2.0)
        assert new_w == 200
        assert new_h == 400

    def test_resize_dim_pil_downscale(self):
        """Test réduction d'une image PIL"""
        img = Image.new('RGB', (100, 200))
        new_w, new_h = get_resize_dim_img_pil(img, 0.5)
        assert new_w == 50
        assert new_h == 100

    def test_resize_dim_pil_no_change(self):
        """Test avec ratio 1.0"""
        img = Image.new('RGB', (100, 200))
        new_w, new_h = get_resize_dim_img_pil(img, 1.0)
        assert new_w == 100
        assert new_h == 200


class TestGetResizeDimImgNp:
    def test_resize_dim_np_upscale(self):
        """Test agrandissement d'une image numpy"""
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        new_w, new_h = get_resize_dim_img_np(img, 2.0)
        assert new_w == 200
        assert new_h == 400

    def test_resize_dim_np_downscale(self):
        """Test réduction d'une image numpy"""
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        new_w, new_h = get_resize_dim_img_np(img, 0.5)
        assert new_w == 50
        assert new_h == 100

    def test_resize_dim_np_no_change(self):
        """Test avec ratio 1.0"""
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        new_w, new_h = get_resize_dim_img_np(img, 1.0)
        assert new_w == 100
        assert new_h == 200


class TestResizeImage:
    def test_resize_pil_with_ratio(self):
        """Test redimensionnement PIL avec ratio"""
        img = Image.new('RGB', (100, 200))
        resized = resize_image(img, ratio=0.5)
        assert isinstance(resized, Image.Image)
        assert resized.size == (50, 100)

    def test_resize_numpy_with_ratio(self):
        """Test redimensionnement numpy avec ratio"""
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        resized = resize_image(img, ratio=0.5)
        assert isinstance(resized, np.ndarray)
        assert resized.shape[:2] == (100, 50)

    def test_resize_pil_with_target_size(self):
        """Test redimensionnement PIL avec taille cible"""
        img = Image.new('RGB', (100, 200))
        resized = resize_image(img, target_size=(50, 75))
        assert resized.size == (50, 75)

    def test_resize_numpy_with_target_size(self):
        """Test redimensionnement numpy avec taille cible"""
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        resized = resize_image(img, target_size=(50, 75))
        assert resized.shape[:2] == (75, 50)

    def test_resize_target_size_priority(self):
        """Test que target_size a la priorité sur ratio"""
        img = Image.new('RGB', (100, 200))
        resized = resize_image(img, ratio=2.0, target_size=(30, 40))
        assert resized.size == (30, 40)

    def test_resize_invalid_ratio(self):
        """Test avec ratio invalide"""
        img = Image.new('RGB', (100, 200))
        with pytest.raises(ValueError, match="ratio doit être strictement positif"):
            resize_image(img, ratio=0)
        with pytest.raises(ValueError):
            resize_image(img, ratio=-1)

    def test_resize_invalid_type(self):
        """Test avec type d'image invalide"""
        with pytest.raises(TypeError, match="doit être une instance de PIL.Image.Image ou numpy.ndarray"):
            resize_image("invalid", ratio=0.5)

    def test_resize_upscale_uses_correct_interpolation(self):
        """Test que l'interpolation correcte est utilisée pour l'agrandissement"""
        img = Image.new('RGB', (10, 10))
        resized = resize_image(img, ratio=2.0)
        assert resized.size == (20, 20)

    def test_resize_downscale_uses_correct_interpolation(self):
        """Test que l'interpolation correcte est utilisée pour la réduction"""
        img = Image.new('RGB', (100, 100))
        resized = resize_image(img, ratio=0.5)
        assert resized.size == (50, 50)


class TestConstants:
    def test_extension_img_constant(self):
        """Vérifie que la constante EXTENSION_IMG contient les bonnes extensions"""
        assert ".jpg" in EXTENSION_IMG
        assert ".jpeg" in EXTENSION_IMG
        assert ".png" in EXTENSION_IMG
        assert ".bmp" in EXTENSION_IMG
        assert ".webp" in EXTENSION_IMG
        assert len(EXTENSION_IMG) == 5
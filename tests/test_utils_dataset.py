# tests/test_utils_dataset.py

import pytest
from stratadl.core.utils.dataset import split_dataset


class TestSplitDataset:
    def test_split_dataset_default_ratios(self):
        """Test avec les ratios par défaut"""
        images = list(range(100))
        train, val, test = split_dataset(images)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100

    def test_split_dataset_custom_ratios(self):
        """Test avec des ratios personnalisés"""
        images = list(range(100))
        train, val, test = split_dataset(images, ratios=(0.6, 0.2, 0.2))

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_dataset_no_overlap(self):
        """Test qu'il n'y a pas de chevauchement entre les splits"""
        images = list(range(100))
        train, val, test = split_dataset(images)

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_dataset_reproducibility(self):
        """Test que le même seed produit les mêmes résultats"""
        images = list(range(100))
        train1, val1, test1 = split_dataset(images, seed=42)
        train2, val2, test2 = split_dataset(images, seed=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_split_dataset_different_seeds(self):
        """Test que des seeds différents produisent des résultats différents"""
        images = list(range(100))
        train1, val1, test1 = split_dataset(images, seed=42)
        train2, val2, test2 = split_dataset(images, seed=123)

        assert train1 != train2

    def test_split_dataset_invalid_ratios(self):
        """Test avec des ratios invalides"""
        images = list(range(100))

        with pytest.raises(AssertionError, match="ratios doivent faire 1.0"):
            split_dataset(images, ratios=(0.5, 0.3, 0.1))

    def test_split_dataset_small_dataset(self):
        """Test avec un petit dataset"""
        images = list(range(10))
        train, val, test = split_dataset(images)

        assert len(train) == 7
        assert len(val) + len(test) == 3

    def test_split_dataset_string_items(self):
        """Test avec des chemins de fichiers"""
        images = [f"image_{i}.jpg" for i in range(100)]
        train, val, test = split_dataset(images)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert all(isinstance(x, str) for x in train)

    def test_split_dataset_extreme_ratios(self):
        """Test avec des ratios extrêmes"""
        images = list(range(100))
        train, val, test = split_dataset(images, ratios=(0.98, 0.01, 0.01))

        assert len(train) == 98
        # val et test peuvent être 1 ou 2 selon l'arrondi
        assert len(val) + len(test) == 2

    def test_split_dataset_all_elements_present(self):
        """Test que tous les éléments sont présents dans un des splits"""
        images = list(range(100))
        train, val, test = split_dataset(images)

        all_elements = set(train + val + test)
        assert all_elements == set(images)
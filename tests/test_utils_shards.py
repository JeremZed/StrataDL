import pytest
import os
import tarfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

# Imports du module à tester
from stratadl.core.utils.shards import (
    WebDatasetConfig,
    expand_url_pattern,
    get_all_shards,
    count_shards,
    create_base_dataset,
    make_dataloader,
    count_images_in_shard,
    compute_data_statistics,
    verify_data_range,
    analyze_shard,
    verify_distribution,
    write_shards,
    create_shards_from_directory
)


class TestExpandUrlPattern:
    """Tests pour la fonction expand_url_pattern"""

    def test_simple_range(self):
        pattern = "https://s.com/f-{00..02}.tar"
        result = expand_url_pattern(pattern)
        expected = [
            'https://s.com/f-00.tar',
            'https://s.com/f-01.tar',
            'https://s.com/f-02.tar'
        ]
        assert result == expected

    def test_range_with_step(self):
        pattern = "https://s.com/shard-{000..010..2}.tar"
        result = expand_url_pattern(pattern)
        assert len(result) == 6
        assert result[0] == "https://s.com/shard-000.tar"
        assert result[-1] == "https://s.com/shard-010.tar"

    def test_no_pattern(self):
        pattern = "https://s.com/single-shard.tar"
        result = expand_url_pattern(pattern)
        assert result == [pattern]

    def test_padding_preservation(self):
        pattern = "file-{0000..0002}.tar"
        result = expand_url_pattern(pattern)
        assert result[0] == "file-0000.tar"
        assert result[1] == "file-0001.tar"
        assert result[2] == "file-0002.tar"

    def test_large_range(self):
        pattern = "shard-{00..99}.tar"
        result = expand_url_pattern(pattern)
        assert len(result) == 100
        assert result[0] == "shard-00.tar"
        assert result[99] == "shard-99.tar"


class TestGetAllShards:
    """Tests pour la fonction get_all_shards"""

    def test_with_list_input(self):
        shards_list = ["shard1.tar", "shard2.tar"]
        result = get_all_shards(shards_list)
        assert result == shards_list

    def test_empty_list_raises_error(self):
        with pytest.raises(ValueError, match="La liste de shards est vide"):
            get_all_shards([])

    def test_http_url_expansion(self):
        pattern = "https://server.com/data-{0..2}.tar"
        result = get_all_shards(pattern)
        assert len(result) == 3
        assert all(url.startswith("https://") for url in result)

    def test_s3_url(self):
        pattern = "s3://bucket/shard-{00..01}.tar"
        result = get_all_shards(pattern)
        assert len(result) == 2
        assert all(url.startswith("s3://") for url in result)

    def test_local_glob_pattern(self, tmp_path):
        # Crée des fichiers de test
        for i in range(3):
            (tmp_path / f"shard-{i}.tar").touch()

        pattern = str(tmp_path / "shard-*.tar")
        result = get_all_shards(pattern)
        assert len(result) == 3

    def test_local_no_match_raises_error(self, tmp_path):
        pattern = str(tmp_path / "nonexistent-*.tar")
        with pytest.raises(ValueError, match="Aucun shard trouvé"):
            get_all_shards(pattern)


class TestCountShards:
    """Tests pour la fonction count_shards"""

    def test_count_with_list(self):
        shards = ["s1.tar", "s2.tar", "s3.tar"]
        assert count_shards(shards) == 3

    def test_count_with_url_pattern(self):
        pattern = "https://server.com/shard-{0..9}.tar"
        assert count_shards(pattern) == 10

    def test_count_local_files(self, tmp_path):
        for i in range(5):
            (tmp_path / f"shard-{i}.tar").touch()

        pattern = str(tmp_path / "shard-*.tar")
        assert count_shards(pattern) == 5


class TestComputeDataStatistics:
    """Tests pour la fonction compute_data_statistics"""

    def test_basic_statistics(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_data_statistics(data)

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert abs(stats["std"] - np.std(data)) < 1e-6

    def test_with_zeros(self):
        data = np.zeros(10)
        stats = compute_data_statistics(data)

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0

    def test_with_negative_values(self):
        data = np.array([-5.0, -3.0, 0.0, 3.0, 5.0])
        stats = compute_data_statistics(data)

        assert stats["min"] == -5.0
        assert stats["max"] == 5.0


class TestCountImagesInShard:
    """Tests pour la fonction count_images_in_shard"""

    def test_count_jpg_images(self, tmp_path):
        # Crée un shard de test
        shard_path = tmp_path / "test.tar"
        with tarfile.open(shard_path, "w") as tar:
            for i in range(5):
                # Crée des fichiers temporaires
                img_path = tmp_path / f"img{i}.jpg"
                img_path.write_bytes(b"fake image data")
                tar.add(img_path, arcname=f"img{i}.jpg")

        count = count_images_in_shard(str(shard_path))
        assert count == 5

    def test_count_with_different_extensions(self, tmp_path):
        shard_path = tmp_path / "test.tar"
        with tarfile.open(shard_path, "w") as tar:
            for ext in [".jpg", ".png", ".txt"]:
                file_path = tmp_path / f"file{ext}"
                file_path.write_bytes(b"data")
                tar.add(file_path, arcname=f"file{ext}")

        count = count_images_in_shard(str(shard_path), ext=".jpg")
        assert count == 1

    def test_remote_url_returns_negative_one(self):
        url = "https://example.com/shard.tar"
        count = count_images_in_shard(url)
        assert count == -1


class TestWriteShards:
    """Tests pour la fonction write_shards"""

    def test_basic_shard_creation(self, tmp_path):
        # Crée des images de test
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_files = []
        for i in range(5):
            img_path = img_dir / f"image_{i}.png"
            # Crée une petite image
            img = Image.new('RGB', (10, 10))
            img.save(img_path)
            image_files.append(str(img_path))

        out_dir = tmp_path / "shards"
        write_shards(image_files, "test", str(out_dir), max_size=1)

        # Vérifie que des shards ont été créés
        shards = list(out_dir.glob("test-*.tar"))
        assert len(shards) > 0

    def test_shard_naming(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        img_path = img_dir / "test.png"
        img = Image.new('RGB', (10, 10))
        img.save(img_path)

        out_dir = tmp_path / "shards"
        write_shards([str(img_path)], "train", str(out_dir), max_size=100)

        shard_files = list(out_dir.glob("train-*.tar"))
        assert len(shard_files) == 1
        assert shard_files[0].name == "train-000000.tar"

    def test_multiple_shards_created(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_files = []
        for i in range(10):
            img_path = img_dir / f"image_{i}.png"
            img = Image.new('RGB', (100, 100))
            img.save(img_path)
            image_files.append(str(img_path))

        out_dir = tmp_path / "shards"
        # Petite taille pour forcer plusieurs shards
        write_shards(image_files, "test", str(out_dir), max_size=0.001)

        shards = sorted(out_dir.glob("test-*.tar"))
        assert len(shards) > 1

    def test_files_in_shard_have_jpg_extension(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        img_path = img_dir / "test.png"
        img = Image.new('RGB', (10, 10))
        img.save(img_path)

        out_dir = tmp_path / "shards"
        write_shards([str(img_path)], "test", str(out_dir), max_size=100)

        shard_path = list(out_dir.glob("test-*.tar"))[0]
        with tarfile.open(shard_path, "r") as tar:
            names = tar.getnames()
            assert all(name.endswith('.jpg') for name in names)


class TestCreateShardsFromDirectory:
    """Tests pour la fonction create_shards_from_directory"""

    @patch('stratadl.core.utils.shards.get_files')
    @patch('stratadl.core.utils.shards.split_dataset')
    @patch('stratadl.core.utils.shards.write_shards')
    def test_creates_three_splits(self, mock_write, mock_split, mock_get_files, tmp_path):
        # Mock des fichiers
        mock_files = [f"img_{i}.jpg" for i in range(100)]
        mock_get_files.return_value = mock_files

        # Mock du split
        mock_split.return_value = (
            mock_files[:70],
            mock_files[70:85],
            mock_files[85:]
        )

        out_dir = str(tmp_path / "shards")
        create_shards_from_directory(
            "input_dir",
            out_dir,
            ratios=(0.7, 0.15, 0.15)
        )

        # Vérifie que write_shards a été appelé 3 fois
        assert mock_write.call_count == 3

        # Vérifie les noms des splits
        calls = mock_write.call_args_list
        split_names = [call[0][1] for call in calls]
        assert "train" in split_names
        assert "val" in split_names
        assert "test" in split_names

    @patch('stratadl.core.utils.shards.get_files')
    @patch('stratadl.core.utils.shards.split_dataset')
    @patch('stratadl.core.utils.shards.write_shards')
    def test_respects_ratios(self, mock_write, mock_split, mock_get_files, tmp_path):
        mock_files = [f"img_{i}.jpg" for i in range(100)]
        mock_get_files.return_value = mock_files

        train_size = 80
        val_size = 10
        test_size = 10

        mock_split.return_value = (
            mock_files[:train_size],
            mock_files[train_size:train_size+val_size],
            mock_files[train_size+val_size:]
        )

        create_shards_from_directory(
            "input_dir",
            str(tmp_path),
            ratios=(0.8, 0.1, 0.1)
        )

        # Vérifie les tailles passées à write_shards
        calls = mock_write.call_args_list
        train_call = next(c for c in calls if c[0][1] == "train")
        val_call = next(c for c in calls if c[0][1] == "val")
        test_call = next(c for c in calls if c[0][1] == "test")

        assert len(train_call[0][0]) == train_size
        assert len(val_call[0][0]) == val_size
        assert len(test_call[0][0]) == test_size


class TestMakeDataloader:
    """Tests pour la fonction make_dataloader"""

    @patch('stratadl.core.utils.shards.get_all_shards')
    @patch('stratadl.core.utils.shards.wds.WebDataset')
    def test_creates_dataloader(self, mock_wds, mock_get_shards):
        mock_get_shards.return_value = ["shard1.tar", "shard2.tar"]

        # Mock du dataset
        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.decode.return_value = mock_dataset
        mock_dataset.to_tuple.return_value = mock_dataset
        mock_dataset.batched.return_value = mock_dataset
        mock_wds.return_value = mock_dataset

        loader = make_dataloader(
            "pattern",
            batch_size=32,
            num_workers=2
        )

        assert loader is not None
        assert isinstance(loader, torch.utils.data.DataLoader)

    @patch('stratadl.core.utils.shards.get_all_shards')
    @patch('stratadl.core.utils.shards.wds.WebDataset')
    def test_applies_transform(self, mock_wds, mock_get_shards):
        mock_get_shards.return_value = ["shard1.tar"]

        mock_dataset = MagicMock()
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.decode.return_value = mock_dataset
        mock_dataset.to_tuple.return_value = mock_dataset
        mock_dataset.map_tuple.return_value = mock_dataset
        mock_dataset.batched.return_value = mock_dataset
        mock_wds.return_value = mock_dataset

        def transform(x):
            return x * 2

        loader = make_dataloader("pattern", transform=transform)

        # Vérifie que map_tuple a été appelé
        mock_dataset.map_tuple.assert_called_once_with(transform)


class TestVerifyDataRange:
    """Tests pour la fonction verify_data_range"""

    def test_computes_statistics(self):
        # Crée un mock loader avec des données connues
        mock_batch = [(torch.rand(10, 3, 32, 32),)]
        mock_loader = [mock_batch]

        min_val, max_val, mean_val, std_val = verify_data_range(
            mock_loader,
            n_samples=10
        )

        assert isinstance(min_val, float)
        assert isinstance(max_val, float)
        assert isinstance(mean_val, float)
        assert isinstance(std_val, float)
        assert min_val <= max_val

    def test_handles_numpy_arrays(self):
        mock_batch = [(np.random.rand(10, 3, 32, 32),)]
        mock_loader = [mock_batch]

        min_val, max_val, mean_val, std_val = verify_data_range(
            mock_loader,
            n_samples=10
        )

        assert min_val >= 0
        assert max_val <= 1

    def test_raises_on_empty_loader(self):
        mock_loader = []

        with pytest.raises(ValueError, match="Aucun échantillon trouvé"):
            verify_data_range(mock_loader, n_samples=10)


class TestVerifyDistribution:
    """Tests pour la fonction verify_distribution"""

    @patch('stratadl.core.utils.shards.get_all_shards')
    @patch('stratadl.core.utils.shards.analyze_shard')
    def test_creates_dataframe(self, mock_analyze, mock_get_shards, tmp_path):
        mock_get_shards.return_value = ["shard1.tar", "shard2.tar"]

        mock_analyze.side_effect = [
            {
                "shard": "shard1.tar",
                "n_images": 100,
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.2
            },
            {
                "shard": "shard2.tar",
                "n_images": 150,
                "min": 0.1,
                "max": 0.9,
                "mean": 0.45,
                "std": 0.18
            }
        ]

        output_file = tmp_path / "stats.csv"
        df = verify_distribution(
            "pattern",
            save_to=str(output_file)
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 2  # 2 shards + lignes de résumé
        assert output_file.exists()

    @patch('stratadl.core.utils.shards.get_all_shards')
    @patch('stratadl.core.utils.shards.analyze_shard')
    def test_calculates_totals(self, mock_analyze, mock_get_shards, tmp_path):
        mock_get_shards.return_value = ["shard1.tar"]
        mock_analyze.return_value = {
            "shard": "shard1.tar",
            "n_images": 100,
            "min": 0.0,
            "max": 1.0,
            "mean": 0.5,
            "std": 0.2
        }

        output_file = tmp_path / "stats.csv"
        df = verify_distribution("pattern", save_to=str(output_file))

        # Vérifie la ligne total_images
        assert "total_images" in df.index


class TestWebDatasetConfig:
    """Tests pour la classe WebDatasetConfig"""

    def test_default_values(self):
        assert WebDatasetConfig.BATCH_SIZE == 64
        assert WebDatasetConfig.N_SHARDS_SHUFFLE == 100
        assert WebDatasetConfig.N_ITEMS_SHUFFLE == 1000
        assert WebDatasetConfig.NUM_WORKERS == 4
        assert WebDatasetConfig.IMAGE_EXT == ".jpg"
        assert WebDatasetConfig.DECODER == "pil"


# Fixtures partagées
@pytest.fixture
def sample_images(tmp_path):
    """Crée des images de test"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    images = []
    for i in range(10):
        img_path = img_dir / f"img_{i}.jpg"
        img = Image.new('RGB', (32, 32), color=(i*10, i*10, i*10))
        img.save(img_path)
        images.append(str(img_path))

    return images


@pytest.fixture
def sample_shard(tmp_path, sample_images):
    """Crée un shard de test"""
    shard_path = tmp_path / "test_shard.tar"

    with tarfile.open(shard_path, "w") as tar:
        for img_path in sample_images:
            tar.add(img_path, arcname=os.path.basename(img_path))

    return str(shard_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
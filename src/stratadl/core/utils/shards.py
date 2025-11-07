from typing import List, Optional, Callable, Dict, Tuple
import re
from glob import glob
import webdataset as wds
import torch
import tarfile
import logging
import numpy as np
import pandas as pd
import os

from stratadl.core.utils.file import get_files
from stratadl.core.utils.dataset import split_dataset
from stratadl.core.preprocessing.image import EXTENSION_IMG

class WebDatasetConfig:
    """Configuration pour les paramètres par défaut de shards d'images."""
    BATCH_SIZE = 64
    N_SHARDS_SHUFFLE = 100
    N_ITEMS_SHUFFLE = 1000
    NUM_WORKERS = 4
    IMAGE_EXT = ".jpg"
    DECODER = "pil"

def expand_url_pattern(pattern: str) -> List[str]:
    """
    Expanse un pattern URL avec accolades en liste d'URLs.
    Supporte le format {start..end} ou {start..end..step}

    Args:
        pattern: Pattern avec accolades, ex: "https://server.com/shard-{000..099}.tar"

    Returns:
        Liste d'URLs expandées

    Examples:
        >>> expand_url_pattern("https://s.com/f-{00..02}.tar")
        ['https://s.com/f-00.tar', 'https://s.com/f-01.tar', 'https://s.com/f-02.tar']
    """
    # Recherche du pattern {start..end} ou {start..end..step}
    match = re.search(r'\{(\d+)\.\.(\d+)(?:\.\.(\d+))?\}', pattern)

    if not match:
        # Pas de pattern trouvé, retourner tel quel dans une liste
        return [pattern]

    start_str, end_str, step_str = match.groups()
    start = int(start_str)
    end = int(end_str)
    step = int(step_str) if step_str else 1

    # Déterminer le padding (nombre de zéros)
    padding = len(start_str)

    # Générer toutes les URLs
    urls = []
    for i in range(start, end + 1, step):
        # Formater avec le bon padding
        num_str = str(i).zfill(padding)
        url = pattern[:match.start()] + num_str + pattern[match.end():]
        urls.append(url)

    return urls


def get_all_shards(shards_pattern: str | List[str]) -> List[str]:
    """
    Retourne la liste des shards (locaux ou distants).

    Args:
        shards_pattern: Pattern glob local, URL distante avec accolades, ou liste d'URLs

    Returns:
        Liste de chemins/URLs

    Raises:
        ValueError: Si aucun shard n'est trouvé (cas local uniquement)
    """
    # Si c'est déjà une liste, on la retourne directement
    if isinstance(shards_pattern, list):
        if not shards_pattern:
            raise ValueError("La liste de shards est vide")
        return shards_pattern

    # Si c'est une URL distante
    if isinstance(shards_pattern, str) and any(
        shards_pattern.startswith(p) for p in ['http://', 'https://', 's3://', 'gs://']
    ):
        # Expanse le pattern avec accolades
        return expand_url_pattern(shards_pattern)

    # Cas local : utiliser glob
    all_shards = sorted(glob(shards_pattern))

    if not all_shards:
        raise ValueError(f"Aucun shard trouvé avec le pattern: {shards_pattern}")

    return all_shards

def count_shards(shards_pattern: str | List[str]) -> int:
    """
    Permet de retourner le nombre total de shards à traiter.

    Args:
        shards_pattern: Pattern glob local, URL distante avec accolades, ou liste d'URLs

    Returns:
        Nombre de shards
    """
    all_shards = get_all_shards(shards_pattern)
    return len(all_shards)


def create_base_dataset(
    shards: List[str],
    shardshuffle: int,
    n_items_shuffle: int,
    image_key: str = "jpg"
) -> wds.WebDataset:
    """
    Crée un dataset WebDataset de base avec les transformations communes.

    Args:
        shards: Liste des chemins vers les shards
        shardshuffle: Nombre de shards à mélanger en mémoire
        n_items_shuffle: Nombre d'items à mélanger par lot
        image_key: Clé pour extraire les images du dataset

    Returns:
        Dataset WebDataset configuré
    """
    return (
        wds.WebDataset(shards, shardshuffle=shardshuffle, empty_check=False)
        .shuffle(n_items_shuffle)
        .decode(WebDatasetConfig.DECODER)
        .to_tuple(image_key)
    )


def make_dataloader(
    pattern: str | List[str],
    batch_size: int = WebDatasetConfig.BATCH_SIZE,
    n_shards_shuffle: int = WebDatasetConfig.N_SHARDS_SHUFFLE,
    n_items_shuffle: int = WebDatasetConfig.N_ITEMS_SHUFFLE,
    transform: Optional[Callable] = None,
    num_workers: int = WebDatasetConfig.NUM_WORKERS
) -> torch.utils.data.DataLoader:
    """
    Crée un DataLoader optimisé pour WebDataset.

    Args:
        pattern: Pattern glob local, URL distante avec accolades, ou liste d'URLs
        batch_size: Taille des lots
        n_shards_shuffle: Nombre de shards à mélanger en mémoire
        n_items_shuffle: Nombre d'items à mélanger
        transform: Fonction de transformation optionnelle
        num_workers: Nombre de workers pour le chargement

    Returns:
        DataLoader configuré

    Raises:
        ValueError: Si aucun shard n'est trouvé
    """
    shards = get_all_shards(pattern)

    dataset = create_base_dataset(shards, n_shards_shuffle, n_items_shuffle)

    if transform is not None:
        dataset = dataset.map_tuple(transform)

    dataset = dataset.batched(batch_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers
    )


def count_images_in_shard(
    shard_path: str,
    ext: str = WebDatasetConfig.IMAGE_EXT
) -> int:
    """
    Compte le nombre d'images dans un shard tar.

    Note: Cette fonction fonctionne uniquement avec des fichiers locaux.
    Pour des URLs distantes, elle tentera d'ouvrir l'URL mais cela peut échouer.

    Args:
        shard_path: Chemin vers le fichier shard (local ou URL)
        ext: Extension des fichiers à compter

    Returns:
        Nombre d'images trouvées
    """
    try:
        with tarfile.open(shard_path, "r") as tar:
            count = sum(1 for member in tar.getmembers() if member.name.endswith(ext))
        return count
    except Exception as e:
        # Si c'est une URL distante, on ne peut pas compter facilement
        # On retourne -1 pour indiquer que le comptage n'est pas disponible
        if any(shard_path.startswith(p) for p in ['http://', 'https://', 's3://', 'gs://']):
            logging.debug(f"Avertissement: Impossible de compter les images pour {shard_path} (URL distante)")
            return -1
        raise e

def compute_data_statistics(
    data: np.ndarray
) -> Dict[str, float]:
    """
    Calcule les statistiques d'un batch de données.

    Args:
        data: Tableau numpy de données

    Returns:
        Dictionnaire avec min, max, mean, std
    """
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std())
    }


def verify_data_range(
    loader: torch.utils.data.DataLoader,
    n_samples: int = 100,
) -> Tuple[float, float, float, float]:
    """
    Vérifie la plage de valeurs des données en parcourant plusieurs batches.

    Args:
        loader: DataLoader à analyser
        n_samples: Nombre total d'échantillons à analyser
        verbose: Afficher les statistiques

    Returns:
        Tuple (min, max, mean, std)
    """
    all_values = []
    samples_collected = 0
    shapes = set()

    for batch in loader:
        # On récupères les images (64 ce qui correspond à la valeur du batch) depuis la liste
        images = batch[0]

        # Conversion en numpy si nécessaire
        if isinstance(images, torch.Tensor):
            images = images.numpy()
        elif isinstance(images, list):
            images = np.array(images)

        # Ajouter les échantillons
        remaining = n_samples - samples_collected
        samples_to_add = images[:remaining]

        # Aplatir complètement chaque image et ajouter les valeurs
        for img in samples_to_add:
            shapes.add(img.shape)
            all_values.extend(img.flatten())

        samples_collected += len(samples_to_add)

        if samples_collected >= n_samples:
            break

    if not all_values:
        raise ValueError("Aucun échantillon trouvé dans le loader")

    # Convertir en array numpy
    data = np.array(all_values)

    stats = compute_data_statistics(data)

    logging.debug(f"\nPlage de données (sur {samples_collected} échantillons):")
    logging.debug(f"   Min:  {stats['min']:.4f}")
    logging.debug(f"   Max:  {stats['max']:.4f}")
    logging.debug(f"   Mean: {stats['mean']:.4f}")
    logging.debug(f"   Std:  {stats['std']:.4f}")

    logging.debug(f"\nListe des dimensions des images:")
    for s in shapes:
        logging.debug(f"   Shape:  {s}")

    return stats['min'], stats['max'], stats['mean'], stats['std']

def analyze_shard(
    shard_path: str,
    batch_size: int,
    transform: Optional[Callable],
    n_samples: int = 10000
) -> Dict[str, any]:
    """
    Analyse un shard et retourne ses statistiques.

    Args:
        shard_path: Chemin vers le shard (local ou URL)
        batch_size: Taille des lots
        transform: Fonction de transformation optionnelle
        n_samples: Nombre d'échantillons pour les statistiques
        verbose: Afficher les détails de l'analyse

    Returns:
        Dictionnaire avec les statistiques du shard
    """
    dataset = create_base_dataset([shard_path], shardshuffle=False, n_items_shuffle=0)

    if transform is not None:
        dataset = dataset.map_tuple(transform)

    dataset = dataset.batched(batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)

    n_images = count_images_in_shard(shard_path)
    min_val, max_val, mean_val, std_val = verify_data_range(loader, n_samples)

    return {
        "shard": shard_path,
        "n_images": n_images,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val
    }

def verify_distribution(
    shards_pattern: str | List[str],
    batch_size: int = WebDatasetConfig.BATCH_SIZE,
    transform: Optional[Callable] = None,
    save_to: str = "dataset_distribution.csv",
    n_samples_per_shard: int = 10000
) -> pd.DataFrame:
    """
    Vérifie la distribution du jeu de données sur tous les shards.

    Args:
        shards_pattern: Pattern glob local, URL distante avec accolades, ou liste d'URLs
        batch_size: Taille des lots
        transform: Fonction de transformation optionnelle
        save_to: Chemin du fichier CSV de sortie
        n_samples_per_shard: Nombre d'échantillons à analyser par shard
        verbose: Afficher les détails pour chaque shard

    Returns:
        DataFrame avec les statistiques de distribution
    """
    all_shards = get_all_shards(shards_pattern)

    results = [
        analyze_shard(shard, batch_size, transform, n_samples_per_shard)
        for shard in all_shards
    ]

    df = pd.DataFrame(results)

    # Ajout des statistiques agrégées
    total_images = df["n_images"].sum()
    mean_stats = df[["n_images", "min", "max", "mean", "std"]].mean()

    # Création des lignes de résumé
    df.loc["mean"] = mean_stats
    df.loc["total_images", "n_images"] = total_images

    df = df.fillna('')
    df.to_csv(save_to)

    return df

def write_shards(files, split_name, outdir, max_size=100):
    """
    Écrit une liste de fichiers images dans des archives TAR fragmentées (shards).

    Cette fonction permet de diviser un ensemble de fichiers en plusieurs archives TAR
    de taille limitée, utile pour gérer de grands datasets.

    Args:
        files (list): Liste des chemins vers les fichiers images à archiver
        split_name (str): Nom de base pour les shards (ex: "train", "val", "test")
        outdir (str): Répertoire de sortie pour les archives TAR
        max_size (int, optional): Taille maximale en octets par shard.
                                  Par défaut 100 MB (100*1024*1024)

    Comportement:
        - Crée des archives TAR nommées "{split_name}-{index:06d}.tar"
        - Chaque fichier est renommé avec l'extension .jpg dans l'archive
        - Commence un nouveau shard quand la taille cumulée dépasse max_size
    """
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(outdir, exist_ok=True)

    max_size = max_size * 1024 * 1024

    shard_idx = 0        # Index du shard actuel
    tar = None           # Objet tarfile actuel
    current_size = 0     # Taille cumulée du shard actuel en octets

    def new_shard():
        """Ferme le shard actuel et en crée un nouveau."""
        nonlocal tar, shard_idx, current_size

        # Ferme le shard précédent s'il existe
        if tar is not None:
            tar.close()

        # Crée le chemin du nouveau shard avec un index formaté sur 6 chiffres
        shard_path = os.path.join(outdir, f"{split_name}-{shard_idx:06d}.tar")
        tar = tarfile.open(shard_path, "w")
        shard_idx += 1
        current_size = 0
        return tar

    # Initialise le premier shard
    tar = new_shard()

    # Parcourt tous les fichiers à archiver
    for img_path in files:
        # Extrait le nom de base sans extension
        base = os.path.basename(img_path).rsplit(".", 1)[0]

        # Crée une entrée TAR avec l'extension .jpg
        info = tarfile.TarInfo(name=f"{base}.jpg")
        file_size = os.path.getsize(img_path)
        info.size = file_size

        # Ajoute le fichier au shard actuel
        with open(img_path, "rb") as f:
            tar.addfile(info, f)

        # Met à jour la taille cumulée
        current_size += file_size

        # Si la taille maximale est atteinte, crée un nouveau shard
        if current_size >= max_size:
            tar = new_shard()

    # Ferme le dernier shard
    if tar is not None:
        tar.close()

    logging.debug(f"Shards {split_name} créés dans {outdir} ({shard_idx} .tar)")


def create_shards_from_directory(dir_input, wds_outdir, ratios = (0.7, 0.15, 0.15), random_state=42, shards_max_size=100):
    """
        Permet de créer l'ensemble des shards (train, val, test) à partir d'un dossier
    """
    images = get_files( input_folder= dir_input, extensions= EXTENSION_IMG)

    dataset_train_size, dataset_val_size, dataset_test_size = ratios

    train_imgs, val_imgs, test_imgs = split_dataset(images, ratios=(dataset_train_size, dataset_val_size, dataset_test_size), seed=random_state)
    logging.debug(f"Taille du Train dataset : {len(train_imgs)}")
    logging.debug(f"Taille du Validation dataset : {len(val_imgs)}")
    logging.debug(f"Taille du Test dataset : {len(test_imgs)}")

    write_shards(train_imgs, "train", wds_outdir, max_size=shards_max_size)
    write_shards(val_imgs, "val", wds_outdir, max_size=shards_max_size)
    write_shards(test_imgs, "test", wds_outdir, max_size=shards_max_size)
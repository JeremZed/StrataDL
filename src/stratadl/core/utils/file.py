from pathlib import Path

import os
import hashlib

import logging

from pathlib import Path
from typing import List, Set

def get_files(input_folder: str, extensions: Set[str] = {'.txt'}, recursive: bool = True) -> List[str]:
    """
    Retourne une liste de fichiers dans le dossier donné.

    Args:
        input_folder (str): Dossier à explorer.
        extensions (Set[str]): Extensions à filtrer (ex: {'.txt', '.jpg'}).
        recursive (bool): Si True, explore tous les sous-dossiers.

    Returns:
        List[str]: Liste des chemins de fichiers correspondant aux extensions.
    """
    input_path = Path(input_folder)
    pattern = '**/*' if recursive else '*'
    paths = [
        str(file) for file in input_path.glob(pattern)
        if file.is_file() and file.suffix.lower() in extensions
    ]
    return paths


def create_directory(path:str) -> None:
    """
    Permet de créer un dossier si celui-ci n'existe pas encore
    """
    os.makedirs(path, exist_ok=True)


def rename_files_with_content_hash(directory, algo="sha1", chunk_size=8192):
    """
    Parcourt récursivement tous les fichiers d'un répertoire et ses sous-répertoires,
    et les renomme avec un hash basé sur le contenu binaire, tout en conservant
    l'extension originale.

    Args:
        directory (str): chemin du répertoire racine
        algo (str): algorithme de hash ('md5', 'sha1', 'sha256', ...)
        chunk_size (int): taille des blocs de lecture pour gros fichiers

    Returns:
        mapping (dict): chemin_complet_ancien -> chemin_complet_nouveau
    """

    logging.debug(f"Scan récursif de: {directory}")
    logging.debug(f"Algorithme: {algo}\n")

    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Algorithme de hash inconnu: {algo}")

    mapping = {}

    # Parcourir récursivement tous les fichiers
    for root, dirs, files in os.walk(directory):
        for fname in files:
            old_path = os.path.join(root, fname)

            # Récupérer extension
            _, ext = os.path.splitext(fname)
            ext = ext.lower()

            # Calculer hash du contenu
            h = hashlib.new(algo)
            try:
                with open(old_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        h.update(chunk)
                digest = h.hexdigest()

                new_name = f"{digest}{ext}"
                new_path = os.path.join(root, new_name)

                # Éviter d'écraser un fichier déjà existant
                if old_path != new_path and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    mapping[old_path] = new_path
                elif old_path != new_path:
                    # Fichier avec le même hash existe déjà
                    logging.debug(f"Collision détectée: {old_path} -> {new_path} (existe déjà)")
                    mapping[old_path] = f"{new_path} (non renommé - doublon)"
                else:
                    # Le fichier a déjà le bon nom
                    mapping[old_path] = new_path

            except (PermissionError, OSError) as e:
                logging.debug(f"Erreur lors du traitement de {old_path}: {e}")
                continue

    renamed = sum(1 for old, new in mapping.items() if old != new and "non renommé" not in new)
    duplicates = sum(1 for v in mapping.values() if "non renommé" in v)
    already_named = len(mapping) - renamed - duplicates

    logging.debug(f"\nTraitement terminé:")
    logging.debug(f"   - {renamed} fichiers renommés")
    logging.debug(f"   - {duplicates} doublons détectés")
    logging.debug(f"   - {already_named} fichiers déjà nommés correctement")
    logging.debug(f"   - {len(mapping)} fichiers traités au total")

    return mapping


import torch
from torchsummary import summary
from pathlib import Path
import pandas as pd
import logging
import json

def get_device():
    """
        Permet de retourner le device "torch" soit utilisation du GPU ou du CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_summary(model, input_size):
    """
        Permet d'afficher la structure globale du model
    """
    summary(model, input_size)


def save_results_to_csv(results, csv_path='results/training_results.csv'):
    """
    Sauvegarde les résultats d'entraînement dans un fichier CSV.
    Ajoute les résultats s'ils existent déjà (et gère le cas d'un fichier vide).

    Args:
        results: Dictionnaire contenant les résultats d'entraînement
        csv_path: Chemin vers le fichier CSV
    """
    # Création du dossier si nécessaire
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Préparation des données pour le CSV
    rows = []
    for config_name, data in results.items():
        row = {'config_name': config_name}

        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                elif isinstance(v, list):
                    items.append((new_key, json.dumps(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        row.update(flatten_dict(data))
        rows.append(row)

    df_new = pd.DataFrame(rows)

    # Lecture robuste du CSV existant
    if path.exists() and path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            # Fichier vide ou corrompu
            df_combined = df_new
    else:
        df_combined = df_new

    # Sauvegarde
    df_combined.to_csv(path, index=False)
    logging.debug(f"✓ Résultats sauvegardés dans {csv_path}")
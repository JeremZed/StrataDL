# StrataDL

Ce dépôt est dédié à l'implémentation de divers algorithmes de Deep Learning. Il inclut également une librairie maison conçue pour simplifier les tâches courantes telles que la création de datasets, la visualisation, et plus encore, spécifiquement pensée pour accélérer les projets d'IA en entreprise.

## Installation

Clonez le dépôt:

```bash
git clone git@github.com:JeremZed/StrataDL.git
cd StrataDL
```

Créez l'environnement et installez les dépendances avec **uv** (https://docs.astral.sh/uv/)

```bash
uv init
uv sync
```

Installer le module en mode développement

```bash
cd package
uv pip install -e .[dev]
```

Lancer les tests unitaires

```bash
pytest -v
```

Couverture du code par les tests

```bash
pytest --cov=stratadl --cov-report=term-missing
```

## Architecture

Le framework repose sur trois couches principales :

- Core : Coeur logique : modèles, agents, RAG, pipelines, tools
- API : Exposition REST via FastAPI
- UI : Interface utilisateur simplifié via un dashboard
- CLI : Interface en ligne de commande pour interagi avec le package
# StrataDL - CLI

## Description

Le sous-module CLI du framework StrataDL fournit une interface en ligne de commande pour interagir avec le package. Vous pouvez l'utilisez pour vérifier la version, lancer le serveur API + dahsboard UI, lancer des traitements etc...

## Utilisation

### Afficher la version

Pour connaître la version encours.

```bash
stratadl --version
```

Affiche:

```bash
StrataDL version 0.0.1
```

### Afficher l'aide

Pour lister l'ensemble des aides.

```bash
stratadl serve --help
```

Affiche:

```bash
Usage: startadl serve [OPTIONS]

  Permet de lancer API + UI

Options:
  --host        TEXT        Adresse IP pour l'API et UI     [default: 127.0.0.1]
  --port-ui     INTEGER     Port UI                         [default: 8500]
  --port-api    INTEGER     Port FastAPI API                [default: 8000]
  --help                    Show this message and exit.
```
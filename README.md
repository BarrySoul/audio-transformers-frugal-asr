# template-ml

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A minimalist project structure based on [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Auteurs

Valentin Rabot, Souleymane Barry et Chebana Sofiane

## Rapport

[Consulter le rapport du projet (PDF)](reports/rapport-projet-audio-transformers.pdf)

## Installation et exécution

Le projet utilise PyTorch. L'entraînement complet est conçu pour un GPU NVIDIA compatible CUDA. Les scripts retombent sur le CPU si CUDA n'est pas disponible, mais l'entraînement peut alors être très lent. Le code ne configure pas actuellement l'accélération Apple MPS.

1. Créer et activer un environnement virtuel Python à la racine du projet :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

2. Sur une machine avec GPU NVIDIA, installer PyTorch et torchaudio avec la commande correspondant au système et à la version CUDA, depuis le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/).
3. Installer les autres dépendances du projet :

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Vérifier la détection du GPU :

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   Pour l'entraînement accéléré par CUDA, cette commande doit afficher `True`.

5. Depuis la racine du projet, lancer les étapes voulues :

   ```bash
   make data    # À exécuter une fois : télécharge et prépare les données
   make train   # Entraîne les modèles
   make test    # Évalue les modèles entraînés
   make bonus1  # Lance l'expérience TTS
   make bonus2  # Lance l'expérience Voxtral
   ```

Les étapes de préparation et d'entraînement téléchargent des données et peuvent demander beaucoup de temps et d'espace disque. `make test` nécessite les données préparées et les checkpoints correspondants.

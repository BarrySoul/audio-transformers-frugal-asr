from pathlib import Path

# Définition de la racine du projet
# Comme config.py est dans src/, le parent du parent est la racine du projet (daaa-project)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Définition des dossiers de données
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Définition du dossier pour sauvegarder les modèles
MODELS_DIR = PROJ_ROOT / "models"

# Création automatique des dossiers s'ils n'existent pas encore
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

"""logger_config = {
    # Vous pouvez ajouter d'autres variables globales ici si besoin
    # comme le taux d'échantillonnage global
    "SAMPLING_RATE": 16000
}"""

SAMPLING_RATE = 16000


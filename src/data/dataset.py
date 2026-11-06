from pathlib import Path
from loguru import logger
import typer
from datasets import load_dataset, Dataset, DatasetDict, Audio
from src.config import RAW_DATA_DIR

app = typer.Typer()

def get_bonus_streaming_dataset(dataset_name="shb777/gemini-flash-2.0-speech", split="en", target_column="kore"):
    """
    Chargement en streaming du dataset audio pour les tâches TTS et Voxtral.
    Caste la colonne cible au format Audio pour un décodage à la volée.
    """
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.cast_column(target_column, Audio(decode=True))
    return ds

@app.command()
def main():
    logger.info("Téléchargement des datasets allégés (Mode Streaming)...")

    # Fluent Speech Commands
    fluent_path = RAW_DATA_DIR / "fluent_speech"
    logger.info("Chargement d'un sous-ensemble de Fluent Speech Commands...")
    ds_fluent_stream = load_dataset("Codec-SUPERB/fluent_speech_commands_synth", streaming=True)
    fluent_split = list(ds_fluent_stream.keys())[0]
    
    logger.info(f"Extraction de 5000 exemples depuis '{fluent_split}' pour le MAE...")
    
    # Générateur pour extraire exactement 5000 audios sans surcharger la mémoire
    def gen_fluent():
        for ex in ds_fluent_stream[fluent_split].take(5000):
            yield ex
            
    ds_fluent = DatasetDict({
        "train": Dataset.from_generator(gen_fluent)
    })
    ds_fluent.save_to_disk(fluent_path)
    
    # Vox Populi (anglais)
    voxpopuli_path = RAW_DATA_DIR / "voxpopuli"
    logger.info("Chargement d'un sous-ensemble de Vox Populi (en)...")
    ds_vox_stream = load_dataset("facebook/voxpopuli", "en", streaming=True)
    
    logger.info("Extraction de 5000 exemples train et 500 exemples test pour l'ASR...")
    
    # Générateur pour extraire 5000 audios pour l'entraînement
    def gen_vox_train():
        for ex in ds_vox_stream["train"].take(5000):
            yield ex
            
    # Générateur pour extraire 500 audios pour l'évaluation (calcul du WER)       
    def gen_vox_test():
        for ex in ds_vox_stream["test"].take(500):
            yield ex

    ds_vox = DatasetDict({
        "train": Dataset.from_generator(gen_vox_train),
        "test": Dataset.from_generator(gen_vox_test)
    })
    ds_vox.save_to_disk(voxpopuli_path)

    logger.success(f"Datasets sauvegardés avec succès dans {RAW_DATA_DIR}")

if __name__ == "__main__":
    app()
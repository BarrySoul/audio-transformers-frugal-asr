import torch
import torchaudio
import numpy as np
from pathlib import Path
from src.config import MODELS_DIR

def save_mel_to_wav(mel_tensor: torch.Tensor, filename: Path, sr: int = 16000, n_fft: int = 1024, hop_length: int = 256):
    """
    Transforme un tenseur de spectrogramme de Mel (Log) en fichier .wav écoutable
    via l'algorithme de Griffin-Lim
    """
    logger.info(f"Traitement du spectrogramme de forme {mel_tensor.shape}...")
    
    # 1. Inverser l'opération log pour revenir à un spectrogramme de Mel linéaire
    # Formule standard: mel = exp(log_mel)
    mel = torch.exp(mel_tensor) - 1e-5 # Petit epsilon de sécurité
    mel = mel.to('cpu')
    
    # Transformateur pour repasser de l'échelle Mel à l'échelle de fréquence linéaire
    inverse_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=64, sample_rate=sr
    )
    
    # Vocodeur Griffin-Lim (power=1.0 signifie qu'on travaille sur la magnitude, pas la puissance)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=hop_length, power=1.0, n_iter=64 # n_iter=64 est un bon compromis qualité/vitesse
    )

    # 3. Appliquer les transformations
    # Torchaudio attend (B, n_freqs, T). On ajoute une dimension Batch et on transpose.
    mel_spec = mel.transpose(0, 1).unsqueeze(0) # (1, 64, T)

    # Passage Mel -> Fréquence linéaire magnitude
    lin_spec = inverse_mel_scale(mel_spec)

    # Reconstruction de phase Griffin-Lim -> Forme d'onde (Waveform)
    waveform = griffin_lim(lin_spec) # (1, 1, samples)

    # 4. Sauvegarder le fichier au format WAV
    torchaudio.save(str(filename), waveform.squeeze(0), sr)
    logger.success(f"Fichier audio généré avec succès dans : {filename}")

if __name__ == "__main__":
    from loguru import logger
    # Charger le spectrogramme prédit généré par le make bonus (juste le premier audio du batch)
    try:
        mel_pred = torch.load(MODELS_DIR / "tts_mel_predicted.pt", map_location='cpu')
        
        # Le tenseur sauvegardé est (B, T, n_mels), on prend le premier audio
        save_mel_to_wav(mel_pred[0], MODELS_DIR / "sortie_synthese.wav")
        
        logger.info("Tu peux maintenant écouter models/sortie_synthese.wav. "
                    "Mais je ne le conseille pas (faut surtout baisser le volume avant haha)")
    except FileNotFoundError:
        logger.error("Fichier models/tts_mel_predicted.pt introuvable. "
                     "As-tu bien lancé 'make bonus' avant ?")
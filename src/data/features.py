import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from loguru import logger
import typer
from datasets import load_from_disk, Audio
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLING_RATE, MODELS_DIR
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=str(MODELS_DIR / 'bpe_50.model'))

def extract_logmel(waveform: torch.Tensor, sr: int, n_mels: int = 64, win_length: int = 400, hop_length: int = 160) -> torch.Tensor:
    """Extract Log-Mel spectrogram from raw audio waveform."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, win_length=win_length, hop_length=hop_length, n_mels=n_mels, power=2.0
    ).to(waveform.device)
    mel_spec = mel_transform(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel_spec)
    return log_mel.transpose(1, 2)

def cmvn(logmel: torch.Tensor) -> torch.Tensor:
    mean = logmel.mean(dim=0, keepdim=True)
    std = logmel.std(dim=0, keepdim=True) + 1e-5
    return (logmel - mean) / std

def apply_spec_augment(logmel: torch.Tensor) -> torch.Tensor:
    """Applique un masquage fréquentiel et temporel (SpecAugment) pour robustifier le modèle."""
    logmel = logmel.unsqueeze(0) 
    freq_masking = T.FrequencyMasking(freq_mask_param=15)
    time_masking = T.TimeMasking(time_mask_param=35)
    
    augmented = time_masking(freq_masking(logmel))
    return augmented.squeeze(0)

app = typer.Typer()

@app.command()
def main():
    logger.info("Génération des features Log-Mel...")
    
    fluent_in = RAW_DATA_DIR / "fluent_speech"
    fluent_out = PROCESSED_DATA_DIR / "fluent_features"
    
    if fluent_in.exists() and not fluent_out.exists():
        logger.info("Traitement de Fluent Speech Commands...")
        ds_fluent = load_from_disk(fluent_in)
        ds_fluent = ds_fluent.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        
        def process_fluent(batch):
            waveform = torch.tensor(batch["audio"]["array"], dtype=torch.float32)
            sr = SAMPLING_RATE
            
            max_len = sr * 10
            if waveform.shape[0] > max_len:
                waveform = waveform[:max_len]
                
            logmel = extract_logmel(waveform.unsqueeze(0), sr).squeeze(0)
            
            logmel = cmvn(logmel) # Normalization

            batch["logmel"] = logmel.numpy().astype(np.float16)
            return batch
        
        ds_fluent = ds_fluent.map(
            process_fluent, 
            remove_columns=["audio"],
            load_from_cache_file=False,
            writer_batch_size=100
        )
        ds_fluent.save_to_disk(fluent_out)
        ds_fluent.cleanup_cache_files()

    vox_in = RAW_DATA_DIR / "voxpopuli"
    vox_out = PROCESSED_DATA_DIR / "voxpopuli_features"
    
    if vox_in.exists() and not vox_out.exists():
        logger.info("Traitement de Vox Populi...")
        ds_vox = load_from_disk(vox_in)
        ds_vox = ds_vox.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        
        def process_vox(batch):
            waveform = torch.tensor(batch["audio"]["array"], dtype=torch.float32)
            sr = SAMPLING_RATE
            
            max_len = sr * 10
            if waveform.shape[0] > max_len:
                waveform = waveform[:max_len]
                
            logmel = extract_logmel(waveform.unsqueeze(0), sr).squeeze(0)
            
            logmel = cmvn(logmel) # Normalization

            batch["logmel"] = logmel.numpy().astype(np.float16)
            batch["logmel_lengths"] = len(logmel)
            
            normalized_text = batch["normalized_text"]
            text_ids = sp.encode(normalized_text)
            batch["labels"] = text_ids
            batch["label_lengths"] = len(text_ids)
            return batch
        
        ds_vox = ds_vox.map(
            process_vox, 
            remove_columns=["audio"],
            load_from_cache_file=False,
            writer_batch_size=100
        )
        ds_vox.save_to_disk(vox_out)
        ds_vox.cleanup_cache_files()
        
    logger.success("Génération des features terminée.")

if __name__ == "__main__":
    app()
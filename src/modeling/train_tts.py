import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from datasets import load_dataset, Audio
from src.config import MODELS_DIR
from src.modeling.models import TextToSpeechTransformer
from src.data.features import extract_logmel, sp
from src.plots import plot_logmel
from src.data.dataset import get_bonus_streaming_dataset
import time

app = typer.Typer()

def prepare_tts_batch(batch_list, device):
    """Extrait le Mel et le Texte à la volée depuis le dataset brut."""
    text_ids_list = []
    mels_list = []
    
    for item in batch_list:
        if "kore" not in item or item["kore"] is None:
            continue
            
        # Traitement du texte
        t_ids = sp.encode(item["text"])
        if len(t_ids) == 0:
            continue
        text_ids_list.append(torch.tensor(t_ids, dtype=torch.long))
        
        # Traitement Audio (voix Kore)
        waveform = torch.tensor(item["kore"]["array"], dtype=torch.float32)
        sr = item["kore"]["sampling_rate"]
        
        # Cropping pour la VRAM
        max_len = sr * 5 # 5 secondes max pour le TTS
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
            
        mel = extract_logmel(waveform.unsqueeze(0), sr).squeeze(0)
        mels_list.append(mel)
        
    if not text_ids_list:
        return None, None

    # Padding
    texts_padded = nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=0).to(device)
    mels_padded = nn.utils.rnn.pad_sequence(mels_list, batch_first=True, padding_value=0.0).to(device)
    
    return texts_padded, mels_padded

@app.command()
def main(
    epochs: int = 20,
    batch_size: int = 16
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Démarrage du Bonus TTS sur {device}...")
    
    # Pour le script TTS (voix "kore") :
    ds_tts = get_bonus_streaming_dataset(target_column="kore")
    data_list = list(ds_tts.take(1500))
    
    # Création de sous-batchs
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    
    # Instanciation du modèle TTS (qui contient maintenant le Post-Net)
    model = TextToSpeechTransformer(vocab_size=50).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss() # MAE Loss utilisée généralement pour les spectrogrammes TTS
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(batches, desc=f"Epoch TTS {epoch+1}/{epochs}")
        
        for b in progress:
            texts, mels = prepare_tts_batch(b, device)
            if texts is None:
                continue
                
            optimizer.zero_grad()
            
            mel_input = mels[:, :-1, :]
            mel_target = mels[:, 1:, :]
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Le modèle amélioré renvoie DEUX sorties
                preds_coarse, preds_refined = model(texts, mel_input)
                
                # On calcule l'erreur pour les deux pour forcer l'encodeur ET le post-net à apprendre
                loss_coarse = criterion(preds_coarse, mel_target)
                loss_refined = criterion(preds_refined, mel_target)
                loss = loss_coarse + loss_refined
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss_refined.item()
            # Et on affiche la loss du Post-Net:
            progress.set_postfix({"l1_loss_postnet": f"{loss_refined.item():.4f}"})
            
    logger.info("Génération de l'image de démonstration...")
    model.eval()

    with torch.no_grad():
        demo_text, demo_mel = prepare_tts_batch([data_list[0]], device)
        if demo_text is not None:

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # On récupère uniquement la sortie raffinée pour l'inférence
            _, pred_mel_refined = model(demo_text, demo_mel[:, :-1, :])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            logger.info(f"Temps de génération TTS pour 1 audio : {inference_time:.4f} secondes")
            
            # Sauvegarde du tenseur pour le script Griffin-Lim
            torch.save(pred_mel_refined, MODELS_DIR / "tts_mel_predicted.pt")
            
            plot_logmel(pred_mel_refined[0], title="Spectrogramme Généré (Post-Net)", save_path=MODELS_DIR / "tts_generated.png")
            plot_logmel(demo_mel[0], title="Spectrogramme Réel (Vérité Terrain)", save_path=MODELS_DIR / "tts_real.png")
            logger.success("Bonus TTS terminé ! Les graphiques ont été sauvegardés dans le dossier models/.")
        else:
            logger.error("Impossible de générer la démo, aucun exemple valide trouvé.")

if __name__ == "__main__":
    app()
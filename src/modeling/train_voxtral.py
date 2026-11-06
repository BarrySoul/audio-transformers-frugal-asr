import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from datasets import load_dataset, Audio

from src.config import MODELS_DIR
from src.modeling.models import AudioTransformerEncoder, VoxtralAudioLLM
from src.data.features import extract_logmel, sp
from src.data.dataset import get_bonus_streaming_dataset

app = typer.Typer()

def prepare_voxtral_batch(batch_list, device):
    text_ids_list = []
    mels_list = []
    
    for item in batch_list:
        if "kore" not in item or item["kore"] is None:
            continue
            
        t_ids = sp.encode(item["text"])
        if len(t_ids) < 2:
            continue
            
        text_ids_list.append(torch.tensor(t_ids, dtype=torch.long))
        
        waveform = torch.tensor(item["kore"]["array"], dtype=torch.float32)
        sr = item["kore"]["sampling_rate"]
        
        max_len = sr * 10
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
            
        mel = extract_logmel(waveform.unsqueeze(0), sr).squeeze(0)
        mels_list.append(mel)
        
    if not text_ids_list:
        return None, None

    texts_padded = nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=0).to(device)
    mels_padded = nn.utils.rnn.pad_sequence(mels_list, batch_first=True, padding_value=0.0).to(device)
    
    return texts_padded, mels_padded

@app.command()
def main(
    epochs: int = 20,
    batch_size: int = 8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Bonus 2 (Voxtral ICLR 2026) on {device}")
    
    ds = get_bonus_streaming_dataset(target_column="kore")
    data_list = list(ds.take(1500))
    
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    
    audio_encoder = AudioTransformerEncoder(
        n_mels=64, dim=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1, patch_size=2, pos_embed="learnable"
    )
    
    model = VoxtralAudioLLM(audio_encoder=audio_encoder, vocab_size=50).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(epochs):
        progress = tqdm(batches, desc=f"Epoch Voxtral {epoch+1}/{epochs}")
        
        for b in progress:
            texts, mels = prepare_voxtral_batch(b, device)
            if texts is None:
                continue
                
            optimizer.zero_grad()
            
            # Décalage classique pour modélisation de langage (LLM) : 
            # On donne les tokens de 0 à N-1, on prédit les tokens de 1 à N
            text_input = texts[:, :-1]
            text_target = texts[:, 1:]
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(mels, text_input)
                loss = criterion(logits.reshape(-1, 50), text_target.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            progress.set_postfix({"ce_loss": f"{loss.item():.4f}"})
            
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "voxtral_model.pt")
    
    # --- PHASE DE TEST (Génération Auto-Régressive) ---
    logger.info("Test d'inférence (génération de texte à partir de l'audio)...")
    model.eval()
    with torch.no_grad():
        demo_text, demo_mel = prepare_voxtral_batch([data_list[0]], device)
        if demo_text is not None:
            # On initie la génération avec juste le tout premier token BPE
            input_ids = demo_text[0:1, :1] # Shape (1, 1)
            
            # Boucle de génération (le LLM prédit le mot suivant, on l'ajoute, etc.)
            generated_ids = input_ids.tolist()[0]
            
            for _ in range(15): # On force la génération de 15 tokens
                inputs = torch.tensor([generated_ids], dtype=torch.long).to(device)
                logits = model(demo_mel[0:1], inputs)
                # Prendre le token avec la plus haute probabilité à la dernière position
                next_token = logits[0, -1, :].argmax().item()
                generated_ids.append(next_token)
                
            predicted_text = sp.decode(generated_ids)
            real_text = sp.decode(demo_text[0].tolist())
            
            logger.info("=========================================")
            logger.info(f"Texte VRAI  : {real_text}")
            logger.info(f"Texte PREDIT: {predicted_text}")
            logger.info("=========================================")
            
    logger.success("Bonus 2 (Voxtral) terminé et testé !")

if __name__ == "__main__":
    app()
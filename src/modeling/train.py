from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch.optim.lr_scheduler import OneCycleLR

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.modeling.models import AudioTransformerEncoder, AudioMAEPretrain

app = typer.Typer()

def make_mae_mask(batch_size: int, seq_len: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(batch_size, seq_len, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.ones([batch_size, seq_len], device=device, dtype=torch.bool)
    for i in range(batch_size):
        mask[i, ids_shuffle[i, :len_keep]] = False
    return mask

def collate_fn(batch):
    logmels = [torch.tensor(item["logmel"]) for item in batch]
    logmels_padded = torch.nn.utils.rnn.pad_sequence(logmels, batch_first=True, padding_value=0.0)
    return {"logmel": logmels_padded}

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "fluent_features",
    model_path: Path = MODELS_DIR / "mae_model.pt",
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 3e-4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_str)
    logger.info(f"Chargement des données depuis {features_path}...")
    
    ds = load_from_disk(features_path)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Initialisation du modèle Professeur pour le MAE
    encoder = AudioTransformerEncoder(n_mels=64, dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, patch_size=2)
    model = AudioMAEPretrain(encoder, n_mels=64, patch_size=2).to(device)

    # Separation of parameters for optimized weight decay
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=lr, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.1
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    criterion = nn.MSELoss()

    logger.info("Début de l'entraînement MAE...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            x_logmel = batch["logmel"].to(device)
            B, T, _ = x_logmel.shape
            
            patch_size = 2
            pad_len = (patch_size - T % patch_size) % patch_size
            if pad_len > 0:
                x_logmel_padded = torch.nn.functional.pad(x_logmel, (0, 0, 0, pad_len))
            else:
                x_logmel_padded = x_logmel
                
            seq_len = x_logmel_padded.shape[1] // patch_size
            mask = make_mae_mask(B, seq_len, mask_ratio=0.75, device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                preds = model(x_logmel_padded, mask)
                targets = x_logmel_padded.view(B, seq_len, -1)
                loss = criterion(preds[mask], targets[mask])
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        logger.info(f"Epoch {epoch+1} terminée. Loss moyenne: {epoch_loss / len(train_loader):.4f}")
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.success(f"Modèle sauvegardé dans {model_path}")

if __name__ == "__main__":
    app()
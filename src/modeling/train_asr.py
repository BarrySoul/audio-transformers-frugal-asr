from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.modeling.models import AudioTransformerEncoder, AudioTransformerCTC
import torchaudio.transforms as T
import random
import numpy as np

app = typer.Typer()

def asr_collate_fn(batch):
    """Padde les spectrogrammes et les labels pour créer un batch uniforme."""
    logmels = []
    labels = []
    logmel_lengths = []
    label_lengths = []
    
    for item in batch:
        logmels.append(torch.tensor(item["logmel"], dtype=torch.float32))
        labels.append(torch.tensor(item["labels"], dtype=torch.long))
        
        logmel_lengths.append(item["logmel_lengths"])
        label_lengths.append(item["label_lengths"])
        
    # Pad sequences : on remplit avec des 0
    # logmels shape: (Batch, Time, Mels)
    logmels_padded = torch.nn.utils.rnn.pad_sequence(logmels, batch_first=True, padding_value=0.0)
    # labels shape: (Batch, Text_Length)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0) # 0 est le blank CTC
    
    return {
        "logmel": logmels_padded,
        "labels": labels_padded,
        "logmel_lengths": torch.tensor(logmel_lengths, dtype=torch.long),
        "label_lengths": torch.tensor(label_lengths, dtype=torch.long)
    }

def distillation_loss(student_logits, teacher_logits, targets, out_lengths, target_lengths, ctc_loss_fn, temperature=2.0, alpha=0.5):
    """Calcule la perte combinée (CTC + Kullback-Leibler Divergence) pour la distillation."""
    hard_loss = ctc_loss_fn(student_logits, targets, out_lengths, target_lengths)
    
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return alpha * hard_loss + (1.0 - alpha) * soft_loss

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "voxpopuli_features",
    pretrained_mae_path: Path = MODELS_DIR / "mae_model.pt",
    model_path: Path = MODELS_DIR / "ctc_model.pt",
    epochs: int = 30,
    batch_size: int = 32,
    vocab_size: int = 50,
    seed: int = 42
):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    final_model_path = model_path.parent / f"{model_path.stem}_seed{seed}{model_path.suffix}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Démarrage du Fine-Tuning ASR avec Distillation (Seed {seed}) sur {device}...")
    
    # Chargement des données
    dataset = load_from_disk(features_path)
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=asr_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialisation du Modèle Professeur (Teacher) - Architecture plus lourde
    teacher_encoder = AudioTransformerEncoder(
        n_mels=64, dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.0, patch_size=2, pos_embed="learnable"
    )
    teacher_model = AudioTransformerCTC(encoder=teacher_encoder, num_classes=vocab_size).to(device)
    
    # Initialisation du Modèle Étudiant (Student) - Architecture frugale pour l'inférence
    student_encoder = AudioTransformerEncoder(
        n_mels=64, dim=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1, patch_size=2, pos_embed="learnable"
    )
    student_model = AudioTransformerCTC(encoder=student_encoder, num_classes=vocab_size).to(device)
    
    # Chargement des poids du pré-entraînement pour le professeur
    use_distillation = False
    if pretrained_mae_path.exists():
        try:
            state_dict = torch.load(pretrained_mae_path, map_location=device)
            encoder_state = {k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")}
            teacher_encoder.load_state_dict(encoder_state)
            teacher_model.eval() # Le professeur ne s'entraîne pas
            for param in teacher_model.parameters():
                param.requires_grad = False
            use_distillation = True
            logger.info("Modèle Professeur chargé. Distillation activée.")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du MAE ({e}). Entraînement classique (sans distillation).")
    else:
        logger.warning(f"Aucun modèle pré-entraîné trouvé à {pretrained_mae_path}. Entraînement classique.")

    # Optimiseur, Scheduler et Loss (uniquement sur l'étudiant)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    accumulation_steps = 4
    total_steps = len(train_loader) * epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=total_steps,
        pct_start=0.15 # Les premiers 15% de l'entraînement servent au Warmup
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 

    # SpecAugment
    freq_masking = T.FrequencyMasking(freq_mask_param=5).to(device)
    time_masking = T.TimeMasking(time_mask_param=10).to(device)

    logger.info("Démarrage de la boucle d'entraînement...")

    for epoch in range(epochs):
        student_model.train()
        
        optimizer.zero_grad()
        
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(progress_bar):
            x_logmel = batch["logmel"].to(device)
            input_lengths = batch["logmel_lengths"].to(device)
            
            targets = batch["labels"].to(device)
            target_lengths = batch["label_lengths"].to(device)
            
            if student_model.training:
                x_logmel = freq_masking(x_logmel)
                x_logmel = time_masking(x_logmel)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                student_logits, out_lengths = student_model(x_logmel, input_lengths)
                student_logits = student_logits.log_softmax(2).transpose(0, 1)
                
                if use_distillation:
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(x_logmel, input_lengths)
                        teacher_logits = teacher_logits.transpose(0, 1)
                    
                    loss = distillation_loss(student_logits, teacher_logits, targets, out_lengths, target_lengths, criterion)
                else:
                    loss = criterion(student_logits, targets, out_lengths, target_lengths)
            
            loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                
                old_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

            real_loss = loss.item() * accumulation_steps
            epoch_loss += real_loss
            progress_bar.set_postfix({"ctc_loss": f"{real_loss:.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} terminée. Loss moyenne: {avg_loss:.4f}")
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(student_model.state_dict(), final_model_path)
    logger.success(f"Modèle ASR Étudiant fine-tuné sauvegardé avec succès dans {final_model_path}")

if __name__ == "__main__":
    app()
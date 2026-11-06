from pathlib import Path
from loguru import logger
import typer
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
from torchmetrics.text import WordErrorRate
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.modeling.models import AudioTransformerEncoder, AudioTransformerCTC
import sentencepiece as spm
import time
import torch.nn.utils.prune as prune

app = typer.Typer()

sp = spm.SentencePieceProcessor(model_file=str(MODELS_DIR / 'bpe_50.model'))

def greedy_decode(logits, blank_id=0):
    pred_ids = torch.argmax(logits, dim=-1) # (B, T)
    decoded_preds = []
    for b in range(pred_ids.size(0)):
        pred_seq = pred_ids[b].tolist()
        filtered = []
        prev_id = -1
        for idx in pred_seq:
            if idx != blank_id and idx != prev_id:
                filtered.append(idx)
            prev_id = idx
        decoded_preds.append(filtered)
    return decoded_preds

def asr_test_collate_fn(batch):
    logmels = []
    logmel_lengths = []
    normalized_texts = []
    
    for item in batch:
        logmels.append(torch.tensor(item["logmel"], dtype=torch.float32))
        logmel_lengths.append(item["logmel_lengths"])
        normalized_texts.append(item["normalized_text"])
        
    logmels_padded = torch.nn.utils.rnn.pad_sequence(logmels, batch_first=True, padding_value=0.0)
    
    return {
        "logmel": logmels_padded,
        "logmel_lengths": torch.tensor(logmel_lengths, dtype=torch.long),
        "normalized_text": normalized_texts
    }

def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    # 1. L1 unstructured pruning appliqué sur TOUTES les couches linéaires du modèle
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prune.remove(module, 'weight')
            
    # 2. Quantification dynamique
    model.eval()
    model.to('cpu')
    
    # On quantifie uniquement la tête de classification CTC pour éviter de faire
    model.classifier = torch.ao.quantization.quantize_dynamic(
        model.classifier, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return model

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "voxpopuli_features",
    model_path: Path = MODELS_DIR / "ctc_model.pt",
    vocab_size: int = 50
):
    # L'inférence frugale quantifiée doit tourner sur CPU
    device = torch.device('cpu')
    logger.info(f"Évaluation du modèle ASR sur {device}")
    
    dataset = load_from_disk(features_path)
    test_loader = DataLoader(dataset["test"], batch_size=4, shuffle=False, collate_fn=asr_test_collate_fn)
    
    wer_metric = WordErrorRate()
    all_wer_scores = []

    seeds = [42, 43, 44, 45, 46]
    
    for run, seed in enumerate(seeds):
        logger.info(f"--- Exécution {run + 1}/5 (Modèle Seed {seed}) ---")
        
        current_model_path = model_path.parent / f"{model_path.stem}_seed{seed}{model_path.suffix}"
        
        if not current_model_path.exists():
            logger.error(f"Le modèle {current_model_path} est introuvable. On passe.")
            continue
            
        # On instancie l'Étudiant frugal (dim=128, depth=4)
        encoder = AudioTransformerEncoder(n_mels=64, dim=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.0, patch_size=2, pos_embed="learnable")
        model = AudioTransformerCTC(encoder=encoder, num_classes=vocab_size)
        
        model.load_state_dict(torch.load(current_model_path, map_location='cpu'))
        
        # Application des optimisations matérielles
        model = optimize_for_inference(model)
        model.eval()

        preds_texts = []
        target_texts = []

        total_inference_time = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x_logmel = batch["logmel"].to(device)
                lengths = batch["logmel_lengths"].to(device)
                ground_truth_texts = batch["normalized_text"]

                start_time = time.perf_counter()
                
                logits, _ = model(x_logmel, lengths)
                decoded_ids = greedy_decode(logits)

                end_time = time.perf_counter()

                total_inference_time += (end_time - start_time)
                total_samples += x_logmel.size(0) 

                batch_preds = [sp.decode(seq) for seq in decoded_ids]
                
                preds_texts.extend(batch_preds)
                target_texts.extend(ground_truth_texts)

        avg_time_per_audio = total_inference_time / total_samples
        
        run_wer = wer_metric(preds_texts, target_texts).item()
        all_wer_scores.append(run_wer)
        
        logger.info(f"Run {run + 1} WER : {run_wer:.4f} | Temps moyen par audio : {avg_time_per_audio:.4f}s")
        
    if len(all_wer_scores) > 0:
        mean_wer = np.mean(all_wer_scores)
        std_wer = np.std(all_wer_scores)
        
        logger.success(f"Résultats finaux (sur {len(all_wer_scores)} runs) : WER = {mean_wer:.4f} ± {std_wer:.4f}")
        
        with open("results.txt", "a") as f:
            f.write(f"Modèle: {model_path.name} | WER: {mean_wer:.4f} ± {std_wer:.4f}\n")
    else:
        logger.error("Aucune évaluation n'a pu être complétée (modèles introuvables).")

if __name__ == "__main__":
    app()
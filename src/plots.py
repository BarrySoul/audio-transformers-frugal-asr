import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

def plot_logmel(logmel, title: str = "Log-Mel Spectrogram", save_path: Path = None):
    """
    Génère et sauvegarde une image 2D d'un spectrogramme Log-Mel.
    """
    if isinstance(logmel, torch.Tensor): # Si c'est un tenseur PyTorch, on le passe en Numpy
        logmel = logmel.detach().cpu().numpy()
        
    if logmel.shape[0] != 64 and logmel.shape[-1] == 64:
        logmel = logmel.T

    plt.figure(figsize=(10, 4))
    plt.imshow(logmel, origin="lower", aspect="auto", cmap="viridis") # origin="lower" permet de mettre les basses fréquences en bas de l'image
    plt.colorbar(label="Log Power (dB)")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    plt.close()
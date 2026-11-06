import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class AudioPatchEmbedding(nn.Module):
    def __init__(self, n_mels: int, dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_mels * patch_size, dim)

    def forward(self, x_logmel: torch.Tensor) -> torch.Tensor:
        B, T, M = x_logmel.shape
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len > 0:
            x_logmel = torch.nn.functional.pad(x_logmel, (0, 0, 0, pad_len))
            T += pad_len
        new_T = T // self.patch_size
        patches = x_logmel.view(B, new_T, self.patch_size * M)
        return self.proj(patches)

class AudioTransformerEncoder(nn.Module):
    def __init__(self, n_mels: int, dim: int, depth: int, num_heads: int, mlp_ratio: float, 
                 dropout: float = 0.1, patch_size: int = 2, max_len: int = 1500, 
                 pos_embed: str = "learnable", use_checkpointing: bool = True):
        super().__init__()
        self.patch_embed = AudioPatchEmbedding(n_mels, dim, patch_size)
        self.use_checkpointing = use_checkpointing
        
        if pos_embed == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        elif pos_embed == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_len)
        else:
            raise ValueError("pos_embed doit être 'learnable' ou 'sinusoidal'")
            
        self.pos_drop = nn.Dropout(p=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=num_heads, 
            dim_feedforward=int(dim * mlp_ratio), 
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.layers = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_logmel: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed(x_logmel)
        
        if isinstance(self.pos_embed, nn.Parameter):
            x = x + self.pos_embed[:, :x.size(1), :]
        else:
            x = self.pos_embed(x)
            
        x = self.pos_drop(x)
        
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        return self.norm(x)

class AudioMAEPretrain(nn.Module):
    def __init__(self, encoder: AudioTransformerEncoder, n_mels: int, patch_size: int = 2):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.norm.normalized_shape[0], n_mels * patch_size)

    def forward(self, x_logmel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x_logmel, mask)
        return self.decoder(features)

class AudioTransformerCTC(nn.Module):
    def __init__(self, encoder: AudioTransformerEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.norm.normalized_shape[0], num_classes)

    def forward(self, x_logmel: torch.Tensor, input_lengths: torch.Tensor = None):
        features = self.encoder(x_logmel)
        logits = self.classifier(features)
        
        if input_lengths is not None:
            # L'encodeur réduit la dimension temporelle lors du patch embedding
            # Il faut donc diviser les longueurs d'entrée par la taille du patch
            patch_size = self.encoder.patch_embed.patch_size
            out_lengths = input_lengths // patch_size
            return logits, out_lengths
            
        # Retourne None pour la longueur si elle n'est pas fournie
        return logits, None

class AudioAdapter(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(audio_dim, text_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# Partie bonus1
class TextToSpeechTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_mels: int = 64):
        super().__init__()
        
        # --- ENCODEUR DE TEXTE ---
        self.text_emb = nn.Embedding(vocab_size, d_model)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.1),
            num_layers=4
        )
        
        self.mel_proj_in = nn.Linear(n_mels, d_model)
        self.audio_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.1),
            num_layers=4
        )
        # Projection intermédiaire grossière vers Mel
        self.mel_proj_out = nn.Linear(d_model, n_mels)
        
        postnet_layers = []
        
        # Première couche de convolution : Mel -> Canal caché
        postnet_layers.append(nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.Tanh(),
            nn.Dropout(0.1)
        ))
        
        # Couches intermédiaires
        for _ in range(3):
            postnet_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.BatchNorm1d(d_model),
                nn.Tanh(),
                nn.Dropout(0.1)
            ))
            
        # Dernière couche : Canal caché -> Mel (sans activation)
        postnet_layers.append(nn.Conv1d(d_model, n_mels, kernel_size=5, padding=2))
        
        self.postnet = nn.Sequential(*postnet_layers)

    def forward(self, texts: torch.Tensor, mel_input: torch.Tensor):
        """
        Passe forward améliorée avec Post-Net résiduel.
        """
        # 1. Traitement du texte
        text_features = self.text_emb(texts)
        memory = self.text_encoder(text_features)
        
        # 2. Préparation de l'audio cible (Teacher Forcing)
        mel_features = self.mel_proj_in(mel_input)
        
        # Masque causal
        seq_len = mel_input.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(mel_input.device)
        
        # 3. Décodage croisé grossier
        decoded = self.audio_decoder(tgt=mel_features, memory=memory, tgt_mask=causal_mask)
        mel_coarse = self.mel_proj_out(decoded) # (B, T, n_mels)
        
        mel_coarse_t = mel_coarse.transpose(1, 2)
        
        # Le Post-Net prédit la CORRECTION résiduelle à ajouter
        mel_residual = self.postnet(mel_coarse_t)
        mel_residual_t = mel_residual.transpose(1, 2)
        
        # Le spectrogramme final est la somme grossier + résiduel
        mel_refined = mel_coarse + mel_residual_t
        
        # Remarque : Pour l'entraînement, on doit calculer la loss sur les deux sorties
        return mel_coarse, mel_refined


# Partie bonus2
class AudioAdapter(nn.Module):
    # Projection de la dimension audio vers la dimension du LLM
    def __init__(self, audio_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(audio_dim, text_dim)
        
    def forward(self, x):
        return self.proj(x)

class VoxtralAudioLLM(nn.Module):
    # Implementation of the ICLR 2026 Paper Audio-LLM architecture
    def __init__(self, audio_encoder, vocab_size: int, audio_dim: int = 128, text_dim: int = 256):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_adapter = AudioAdapter(audio_dim, text_dim)
        self.text_embed = nn.Embedding(vocab_size, text_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=text_dim, nhead=8, batch_first=True)
        self.language_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.lm_head = nn.Linear(text_dim, vocab_size)
        
    def forward(self, x_logmel: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # Extraction des features acoustiques
        audio_features = self.audio_encoder(x_logmel)
        
        # Alignement des dimensions (Cross-Modality Adapter)
        audio_embeds = self.audio_adapter(audio_features)
        
        # Embedding du texte cible (Teacher Forcing)
        text_embeds = self.text_embed(text_ids)
        
        # Generation autoregressive (Causal Mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(text_embeds.size(1)).to(text_embeds.device)
        
        # Cross-attention entre le texte et l'audio
        out = self.language_decoder(tgt=text_embeds, memory=audio_embeds, tgt_mask=tgt_mask)
        return self.lm_head(out)
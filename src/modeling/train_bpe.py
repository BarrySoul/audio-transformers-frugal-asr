import sentencepiece as spm
from src.config import MODELS_DIR

print("Entraînement du Tokenizer BPE en cours...")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

spm.SentencePieceTrainer.train(
    input='corpus.txt', 
    model_prefix=(MODELS_DIR / 'bpe_50'), # Le nom des fichiers de sortie
    vocab_size=50,
    model_type='bpe',
    pad_id=0,
    unk_id=1,
    bos_id=-1,
    eos_id=-1
)
print("Fichiers bpe_50.model et bpe_50.vocab créés.")
from datasets import load_dataset

print("Chargement du dataset VoxPopuli...")
dataset = load_dataset("facebook/voxpopuli", "en", split="train", streaming=True)

dataset = dataset.select_columns(["normalized_text"])

print("Création du fichier corpus.txt...")
with open("corpus.txt", "w", encoding="utf-8") as f:
    # On extrait juste les 2000 premières phrases pour aller vite
    for i, item in enumerate(dataset):
        if i >= 2000:
            break
        f.write(item["normalized_text"] + "\n")

print("Le fichier corpus.txt a été créé à la racine du projet.")
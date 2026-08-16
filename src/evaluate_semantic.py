"""
Evaluates semantic retrieval using Flickr30K's human captions as ground truth queries.
Unlike evaluate_tradeoffs.py which measures how well approximate search copies exact search,
this script measures real retrieval quality: does the system find the right image for a given caption?
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate():
    print("Loading pre-trained CLIP model for text encoding...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    def encode_text(queries):
        inputs = processor(
            text=queries, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(DEVICE)
        with torch.no_grad():
            out = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            features = model.text_projection(out[1])
        emb = features.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb

    captions_path = None
    # Look for results.csv either in DATA_DIR or in typical Kaggle paths if running there
    candidates = [
        DATA_DIR / "results.csv",
        DATA_DIR / "flickr30k_images" / "results.csv",
        Path("/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"),
        Path("/kaggle/input/flickr30k-images/results.csv")
    ]
    for p in candidates:
        if p.exists():
            captions_path = p
            break
            
    if not captions_path:
        print("Captions file (results.csv) not found in data/. Skipping semantic evaluation.")
        print("Please download results.csv from Flickr30K and place it in the data/ directory.")
        return

    print(f"Found captions at: {captions_path}")
    raw_lines = open(captions_path, "r").readlines()

    records = []
    for line in raw_lines[1:]:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            img_name = parts[0].strip()
            caption = parts[2].strip()
            if img_name and caption and img_name.endswith(".jpg"):
                records.append({"image_name": img_name, "caption": caption})

    captions_df = pd.DataFrame(records)
    print(f"Parsed {len(captions_df):,} captions across {captions_df['image_name'].nunique():,} images")

    # Load index paths to map image names to their position in the FAISS index
    paths_file = DATA_DIR / "index_paths.json"
    if not paths_file.exists():
        print(f"Missing {paths_file}! Run feature_extractor.py first.")
        return
        
    with open(paths_file, "r") as f:
        index_paths = json.load(f)
        
    path_to_idx = {os.path.basename(p): i for i, p in enumerate(index_paths)}
    
    captions_df["index_pos"] = captions_df["image_name"].map(path_to_idx)
    valid = captions_df.dropna(subset=["index_pos"]).copy()
    valid["index_pos"] = valid["index_pos"].astype(int)
    print(f"Usable captions (image is in index set): {len(valid):,}")
    
    n_eval = min(2000, len(valid))
    sample = valid.sample(n=n_eval, random_state=42)
    print(f"Sampling {n_eval:,} captions for evaluation...")
    
    caption_texts = sample["caption"].tolist()
    gold = sample["index_pos"].tolist()
    text_embs = []

    for i in tqdm(range(0, len(caption_texts), 64), desc="Encoding captions"):
        batch = caption_texts[i : i + 64]
        text_embs.append(encode_text(batch))

    text_embs = np.concatenate(text_embs, axis=0)
    
    index_files = {
        "FlatIP": DATA_DIR / "index_flat.faiss",
        "IVFFlat": DATA_DIR / "index_ivfflat.faiss",
        "IVFPQ": DATA_DIR / "index_ivfpq.faiss",
    }
    
    print("\n" + "=" * 60)
    print("Caption -> Image Retrieval (Semantic Recall)")
    print("=" * 60)
    print(f"{'Index':<10} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8}")
    print("-" * 45)

    for name, path in index_files.items():
        if not path.exists():
            print(f"Index {name} missing at {path}")
            continue
            
        idx = faiss.read_index(str(path))
        if hasattr(idx, "nprobe"):
            idx.nprobe = 10

        _, I = idx.search(text_embs, 10)

        r = {}
        for k in [1, 5, 10]:
            hits = sum(1 for i, g in enumerate(gold) if g in I[i][:k])
            r[k] = (hits / len(gold)) * 100

        print(f"{name:<10} | {r[1]:<8.2f} | {r[5]:<8.2f} | {r[10]:<8.2f}")

    print("\nThis is the metric that actually matters for a search engine.")
    print("It tells you whether the system retrieves the RIGHT image for a given description.")


if __name__ == "__main__":
    evaluate()

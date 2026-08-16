"""
Here we take the raw Flickr30K images and push them through a frozen CLIP ViT-B/32 model.
It extracts a 512-dimensional vector for each image.
We also handle splitting the dataset into a 90% index set (which we'll search over) 
and a 10% query set (which we'll use to test our recall later).
"""

import json
import os
import random
from pathlib import Path

import torch
import numpy as np
import faiss
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_image_dir():
    """Find a Flickr30K image directory from env var or common Kaggle paths."""
    candidates = [
        os.environ.get("FLICKR_DIR"),
        "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images",
        "/kaggle/input/flickr-image-dataset/flickr30k_images",
        "/kaggle/input/flickr30k-images/flickr30k_images",
        "./data/flickr30k_images",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir() and len(list(path.glob("*.jpg"))) > 100:
            return path

    raise FileNotFoundError(
        "Flickr30K images not found. Set FLICKR_DIR=/path/to/flickr30k_images "
    )


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception:
            # Return None for corrupt images so we can filter them out
            return None

def pil_collate(batch):
    """Custom collate function to drop corrupt (None) images."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def main():
    # 1. Data Loading
    print("Loading pre-trained CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print("Loading Flickr30K image paths...")
    image_dir = find_image_dir()
    image_paths = sorted(image_dir.glob("*.jpg"))
    print(f"Found {len(image_paths):,} images in {image_dir}")

    dataset = ImageFolderDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pil_collate)

    all_embeddings = []
    all_paths = []

    # 2. Feature Extraction
    print("Extracting CLIP features...")
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting"):
            if not images:
                continue
                
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            
            # Call vision_model and visual_projection directly for API stability
            # across different HuggingFace versions.
            vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
            pooled_output = vision_outputs[1]  # pooler_output is the second element
            features = model.visual_projection(pooled_output)
            
            all_embeddings.append(features.cpu().numpy())
            all_paths.extend(paths)

    final_embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    print(f"Extraction complete! Final shape: {final_embeddings.shape}")

    # L2 Normalize so that inner product perfectly matches cosine similarity
    faiss.normalize_L2(final_embeddings)

    # 3. Train / Query Split
    print("Creating 90/10 index/query split...")
    num_total = len(all_paths)
    indices = list(range(num_total))
    random.seed(42)
    random.shuffle(indices)

    split_idx = int(num_total * 0.9)
    train_indices = indices[:split_idx]
    query_indices = indices[split_idx:]

    index_embeddings = final_embeddings[train_indices]
    index_paths = [all_paths[i] for i in train_indices]

    query_embeddings = final_embeddings[query_indices]
    query_paths = [all_paths[i] for i in query_indices]

    print(f"Index set: {len(index_embeddings)} images")
    print(f"Query set: {len(query_embeddings)} images")

    # 4. Save to Disk
    np.save(DATA_DIR / "index_embeddings.npy", index_embeddings)
    with open(DATA_DIR / "index_paths.json", "w") as f:
        json.dump(index_paths, f)

    np.save(DATA_DIR / "query_embeddings.npy", query_embeddings)
    with open(DATA_DIR / "query_paths.json", "w") as f:
        json.dump(query_paths, f)

    print(f"Saved datasets to {DATA_DIR}/")

if __name__ == "__main__":
    main()

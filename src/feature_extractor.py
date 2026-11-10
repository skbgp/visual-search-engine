"""
Here we take the raw Flickr30K images and push them through a frozen CLIP ViT-B/32 model.
It extracts a 512-dimensional vector for each image.
We also handle splitting the dataset into a 90% index set (which we'll search over) 
and a 10% query set (which we'll use to test our recall later).
"""

import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def find_image_dir():
    """Find a Flickr30K image directory from env var or common Kaggle paths."""
    configured = os.environ.get("FLICKR_DIR")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"FLICKR_DIR does not point to a directory: {path}")
        return path

    candidates = [
        DATA_DIR / "flickr30k_images" / "flickr30k_images",
        DATA_DIR / "flickr30k_images",
        "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images",
        "/kaggle/input/flickr-image-dataset/flickr30k_images",
        "/kaggle/input/flickr30k-images/flickr30k_images",
        "/kaggle/input/datasets/hsankesara/flickr-image-dataset/flickr30k_images/flickr30k_images",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir() and any(path.glob("*.jpg")):
            return path

    raise FileNotFoundError(
        "Flickr30K images not found. Set FLICKR_DIR to the folder containing the JPG files."
    )


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as image:
                return image.convert("RGB"), str(path)
        except OSError:
            return None

def pil_collate(batch):
    """Custom collate function to drop corrupt (None) images."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return [], []
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def extract_embeddings(image_paths, model, processor):
    dataset = ImageFolderDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pil_collate)
    all_embeddings = []
    all_paths = []

    with torch.inference_mode():
        for images, paths in tqdm(dataloader, desc="Extracting"):
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
            features = model.visual_projection(vision_outputs.pooler_output)
            all_embeddings.append(features.cpu().numpy())
            all_paths.extend(paths)
    return all_embeddings, all_paths


def save_split(embeddings, paths):
    if len(paths) < 2:
        raise ValueError("Need at least two readable images to split the embeddings.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("CLIP produced an invalid embedding.")
    embeddings /= norms

    print("Creating 90/10 index/query split...")
    indices = np.random.RandomState(42).permutation(len(paths))
    query_count = max(1, len(paths) // 10)
    query_indices = indices[:query_count]
    train_indices = indices[query_count:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, selected in (("index", train_indices), ("query", query_indices)):
        np.save(DATA_DIR / f"{name}_embeddings.npy", embeddings[selected])
        with (DATA_DIR / f"{name}_paths.json").open("w", encoding="utf-8") as file:
            json.dump([paths[i] for i in selected], file)
        print(f"{name.title()} set: {len(selected):,} images")
    print(f"Saved datasets to {DATA_DIR}/")


def main():
    image_dir = find_image_dir()
    image_paths = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two JPG images in {image_dir}.")
    print(f"Found {len(image_paths):,} images in {image_dir}")

    print(f"Loading CLIP for image encoding on {DEVICE}...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    embeddings, paths = extract_embeddings(image_paths, model, processor)
    print(f"Skipped {len(image_paths) - len(paths):,} unreadable images.")
    if not embeddings:
        raise ValueError(f"No readable JPG images in {image_dir}.")
    save_split(np.concatenate(embeddings).astype("float32"), paths)


if __name__ == "__main__":
    main()

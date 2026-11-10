"""CLIP text encoding worker for the caption-to-image benchmark."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def encode_captions(captions):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading CLIP for text encoding on {device}...", flush=True)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(captions), 64), desc="Encoding captions"):
            inputs = processor(
                text=captions[i : i + 64], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            ).to(device)
            output = model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            embeddings.append(model.text_projection(output.pooler_output).cpu().numpy())
    return np.concatenate(embeddings).astype("float32")


if __name__ == "__main__":
    captions_path, embeddings_path = sys.argv[1:]
    captions = json.loads(Path(captions_path).read_text(encoding="utf-8"))
    np.save(embeddings_path, encode_captions(captions))

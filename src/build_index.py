"""
Takes our 512-dimensional CLIP embeddings and builds three different FAISS indices:
1. FlatIP (Exact nearest neighbor, slow but perfect recall)
2. IVFFlat (Approximate, clusters the space to speed up search)
3. IVFPQ (Approximate + Compressed, uses Product Quantization to drastically shrink memory)

This sets us up perfectly to evaluate the memory/latency/recall tradeoffs later.
"""

import os
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "index_embeddings.npy"
DIMENSION = 512

def build_indices():
    print("Loading index embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    num_samples = embeddings.shape[0]
    print(f"Loaded {num_samples} embeddings of dimension {DIMENSION}.")

    # 1. FlatIP (exact search)
    print("\nBuilding FlatIP Index...")
    index_flat = faiss.IndexFlatIP(DIMENSION)
    index_flat.add(embeddings)
    faiss.write_index(index_flat, str(DATA_DIR / "index_flat.faiss"))
    print(f"  -> Saved FlatIP (Total vectors: {index_flat.ntotal})")

    # 2. IVFFlat (clustering)
    print("\nBuilding IVFFlat Index...")
    nlist = 100
    quantizer = faiss.IndexFlatIP(DIMENSION)
    index_ivf = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("  -> Training IVFFlat...")
    index_ivf.train(embeddings)
    print("  -> Adding embeddings...")
    index_ivf.add(embeddings)
    faiss.write_index(index_ivf, str(DATA_DIR / "index_ivfflat.faiss"))
    print(f"  -> Saved IVFFlat (Total vectors: {index_ivf.ntotal})")

    # 3. IVFPQ (clustering + product quantization)
    print("\nBuilding IVFPQ Index...")
    m = 64
    nbits = 8
    quantizer_pq = faiss.IndexFlatIP(DIMENSION)
    index_ivfpq = faiss.IndexIVFPQ(quantizer_pq, DIMENSION, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    
    print("  -> Training IVFPQ...")
    index_ivfpq.train(embeddings)
    print("  -> Adding embeddings...")
    index_ivfpq.add(embeddings)
    faiss.write_index(index_ivfpq, str(DATA_DIR / "index_ivfpq.faiss"))
    print(f"  -> Saved IVFPQ (Total vectors: {index_ivfpq.ntotal})")

    print("\nAll FAISS indices built successfully!")

if __name__ == "__main__":
    if not EMBEDDINGS_PATH.exists():
        print(f"Could not find {EMBEDDINGS_PATH}. Run feature_extractor.py first!")
    else:
        build_indices()

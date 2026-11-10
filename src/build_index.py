"""
Takes our 512-dimensional CLIP embeddings and builds three different FAISS indices:
1. FlatIP (Exact nearest neighbor, slow but perfect recall)
2. IVFFlat (Approximate, clusters the space to speed up search)
3. IVFPQ (Approximate + Compressed, uses Product Quantization to drastically shrink memory)

This sets us up perfectly to evaluate the memory/latency/recall tradeoffs later.
"""

from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "index_embeddings.npy"
NLIST = 100
PQ_SUBVECTORS = 64
PQ_BITS = 8

def build_indices():
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}. Run src.feature_extractor first.")
    print("Loading index embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("Index embeddings must be a finite 2D array with at least one column.")
    num_samples, dimension = embeddings.shape
    min_samples = max(NLIST, 2 ** PQ_BITS)
    if num_samples < min_samples:
        raise ValueError(f"Need at least {min_samples} index vectors to train IVF and PQ, got {num_samples}.")
    if dimension % PQ_SUBVECTORS:
        raise ValueError(f"Embedding dimension must be divisible by {PQ_SUBVECTORS} for PQ.")
    print(f"Loaded {num_samples} embeddings of dimension {dimension}.")

    # 1. FlatIP (exact search)
    print("\nBuilding FlatIP Index...")
    index_flat = faiss.IndexFlatIP(dimension)
    index_flat.add(embeddings)

    # 2. IVFFlat (clustering)
    print("\nBuilding IVFFlat Index...")
    quantizer = faiss.IndexFlatIP(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, NLIST, faiss.METRIC_INNER_PRODUCT)
    
    print("  -> Training IVFFlat...")
    index_ivf.train(embeddings)
    print("  -> Adding embeddings...")
    index_ivf.add(embeddings)

    # 3. IVFPQ (clustering + product quantization)
    print("\nBuilding IVFPQ Index...")
    quantizer_pq = faiss.IndexFlatIP(dimension)
    index_ivfpq = faiss.IndexIVFPQ(
        quantizer_pq, dimension, NLIST, PQ_SUBVECTORS, PQ_BITS, faiss.METRIC_INNER_PRODUCT
    )
    
    print("  -> Training IVFPQ...")
    index_ivfpq.train(embeddings)
    print("  -> Adding embeddings...")
    index_ivfpq.add(embeddings)
    for name, index in (("flat", index_flat), ("ivfflat", index_ivf), ("ivfpq", index_ivfpq)):
        path = DATA_DIR / f"index_{name}.faiss"
        faiss.write_index(index, str(path))
        print(f"Saved {path.name} with {index.ntotal:,} vectors.")

    print("\nAll FAISS indices built successfully!")

if __name__ == "__main__":
    build_indices()

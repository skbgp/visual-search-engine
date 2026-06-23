"""
This is where the magic happens for the ablation study.
We benchmark the FAISS indexes (FlatIP vs IVFFlat vs IVFPQ) against our held-out query embeddings.

We're tracking three things:
1. Recall@10 (does it still find the right images?)
2. Query latency (how much faster is it?)
3. Index size (how much memory did we save?)

We use FlatIP (exact search) as our ground-truth baseline to see how much the approximate methods degrade recall.
"""

import os
import time
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QUERY_EMBEDDINGS_PATH = DATA_DIR / "query_embeddings.npy"
K = 10  # Top-K for Recall@10

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def evaluate():
    print("Loading held-out query embeddings...")
    if not QUERY_EMBEDDINGS_PATH.exists():
        print(f"{QUERY_EMBEDDINGS_PATH} missing! Run feature_extractor.py first.")
        return
        
    queries = np.load(QUERY_EMBEDDINGS_PATH).astype("float32")
    num_queries = queries.shape[0]
    print(f"Loaded {num_queries} queries.")
    
    indices = {
        "FlatIP": DATA_DIR / "index_flat.faiss",
        "IVFFlat": DATA_DIR / "index_ivfflat.faiss",
        "IVFPQ": DATA_DIR / "index_ivfpq.faiss",
    }
    
    results = {}
    ground_truth = []
    
    for name, path in indices.items():
        if not path.exists():
            print(f"{path} missing! Run build_index.py first.")
            return
            
        # 1. Measure Memory 
        size_mb = get_file_size_mb(path)
        
        # Load index
        index = faiss.read_index(str(path))
        if hasattr(index, "nprobe"):
            index.nprobe = 10  # Search 10 clusters out of 100
            
        # 2. Measure Latency
        # Warmup
        index.search(queries[:10], K)
        
        start_time = time.time()
        D, I = index.search(queries, K)
        end_time = time.time()
        
        avg_latency_ms = ((end_time - start_time) / num_queries) * 1000
        
        if name == "FlatIP":
            ground_truth = I
            recall = 100.0
        else:
            # 3. Recall@10 against FlatIP baseline
            correct = 0
            for i in range(num_queries):
                gt_set = set(ground_truth[i])
                pred_set = set(I[i])
                correct += len(gt_set.intersection(pred_set))
            
            recall = (correct / (num_queries * K)) * 100
            
        results[name] = {
            "size_mb": size_mb,
            "latency_ms": avg_latency_ms,
            "recall": recall
        }
        
    # Print the benchmark table
    print("\n" + "=" * 60)
    print("FAISS TRADE-OFF RESULTS (Image-to-Image)")
    print("=" * 60)
    print(f"{'Index':<15} | {'Memory (MB)':<12} | {'Latency (ms)':<15} | {'Recall@10 (%)':<15}")
    print("-" * 65)
    for name, res in results.items():
        print(f"{name:<15} | {res['size_mb']:<12.2f} | {res['latency_ms']:<15.3f} | {res['recall']:<15.2f}")
    
    flat_mem = results["FlatIP"]["size_mb"]
    pq_mem = results["IVFPQ"]["size_mb"]
    
    compression_ratio = flat_mem / pq_mem
    print("\nSummary:")
    print(f"IVFPQ achieved {compression_ratio:.1f}x compression relative to FlatIP.")
    
if __name__ == "__main__":
    evaluate()

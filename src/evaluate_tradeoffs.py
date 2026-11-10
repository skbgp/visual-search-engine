"""Compare approximate indexes against exact nearest-neighbor search."""

import time
from pathlib import Path
from statistics import median

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QUERY_EMBEDDINGS_PATH = DATA_DIR / "query_embeddings.npy"
K = 10
DEFAULT_NPROBE = 10
NPROBE_VALUES = (1, 5, 10, 20, 50)


def search_benchmark(index, queries, k):
    """Return search results and the median batch time divided by query count."""
    index.search(queries, k)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        _, ids = index.search(queries, k)
        times.append(time.perf_counter() - start)
    return ids, 1000 * median(times) / len(queries)


def neighbor_recall(exact_ids, approximate_ids):
    hits = sum(
        len(set(expected).intersection(doc_id for doc_id in actual if doc_id >= 0))
        for expected, actual in zip(exact_ids, approximate_ids)
    )
    return 100 * hits / exact_ids.size


def evaluate():
    if not QUERY_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing {QUERY_EMBEDDINGS_PATH}. Run src.feature_extractor first.")
    queries = np.load(QUERY_EMBEDDINGS_PATH).astype("float32")
    if queries.ndim != 2 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError("Query embeddings must be a nonempty, finite 2D array.")

    index_paths = {
        "FlatIP": DATA_DIR / "index_flat.faiss",
        "IVFFlat": DATA_DIR / "index_ivfflat.faiss",
        "IVFPQ": DATA_DIR / "index_ivfpq.faiss",
    }
    indexes = {}
    for name, path in index_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run src.build_index first.")
        if path.stat().st_mtime_ns < QUERY_EMBEDDINGS_PATH.stat().st_mtime_ns:
            raise ValueError(f"{path.name} predates the current embeddings. Rebuild the indexes.")
        indexes[name] = faiss.read_index(str(path))

    index_size = indexes["FlatIP"].ntotal
    if not index_size or any(
        index.ntotal != index_size or index.d != queries.shape[1]
        for index in indexes.values()
    ):
        raise ValueError("Index sizes or dimensions do not match. Rebuild all three indexes.")
    k = min(K, index_size)
    exact_ids, flat_ms = search_benchmark(indexes["FlatIP"], queries, k)

    print(f"Image-to-image benchmark on {len(queries):,} held-out queries")
    print(f"{'Index':<12} {'File (MiB)':>12} {'Batch ms/query':>16} {f'Recall@{k} (%)':>16}")
    for name, index in indexes.items():
        if name == "FlatIP":
            recall, ms = 100.0, flat_ms
        else:
            index.nprobe = min(DEFAULT_NPROBE, index.nlist)
            ids, ms = search_benchmark(index, queries, k)
            recall = neighbor_recall(exact_ids, ids)
        size_mib = index_paths[name].stat().st_size / 2**20
        print(f"{name:<12} {size_mib:>12.2f} {ms:>16.3f} {recall:>16.2f}")

    ratio = index_paths["FlatIP"].stat().st_size / index_paths["IVFPQ"].stat().st_size
    print(f"FlatIP file size / IVFPQ file size = {ratio:.1f}x")

    print("\nIVFPQ nprobe sweep")
    print(f"{'nprobe':>8} {'Batch ms/query':>16} {f'Recall@{k} (%)':>16}")
    index = indexes["IVFPQ"]
    for nprobe in NPROBE_VALUES:
        if nprobe <= index.nlist:
            index.nprobe = nprobe
            ids, ms = search_benchmark(index, queries, k)
            print(f"{nprobe:>8} {ms:>16.3f} {neighbor_recall(exact_ids, ids):>16.2f}")


if __name__ == "__main__":
    evaluate()

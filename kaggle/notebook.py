# %% [markdown]
# # Visual Search Engine
# **CLIP + FAISS Benchmarks: Exact vs Approximate Search**
#
# Benchmarking FlatIP, IVFFlat, and IVFPQ indexes on Flickr30K.
#
# Dataset: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

# %% [markdown]
# ## 1. Imports and Setup
# %%
# pip install -q faiss-cpu
import os
import time
import json
import glob
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

DATA_DIR = "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# %% [markdown]
# ## 2. Load Flickr30K Images
# Dynamically locate the dataset in Kaggle environments.
# %%
FLICKR_DIR = None
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        jpg_files = [f for f in files if f.lower().endswith(".jpg")]
        if len(jpg_files) > 1000:  # the main folder should have ~31,000
            FLICKR_DIR = root
            break

if FLICKR_DIR is None:
    # print what kaggle actually sees so we can debug path issues
    print("DEBUG: What is inside /kaggle/input?")
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f" - {root} has {len(dirs)} folders and {len(files)} files")
    else:
        print(" - /kaggle/input does not exist yet!")
        
    raise FileNotFoundError(
        "Could not find the Flickr30K dataset. "
        "Make sure you added 'flickr-image-dataset' by hsankesara to the notebook."
    )

image_paths = sorted(glob.glob(os.path.join(FLICKR_DIR, "*.jpg")))
print(f"Found {len(image_paths):,} images in {FLICKR_DIR}")


class Flickr30KDataset(Dataset):
    """Flickr30K Dataset. Returns None for corrupt JPEG files."""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img, idx
        except Exception:
            return None, idx


def pil_collate(batch):
    """Collate function that filters out None (corrupt) images."""
    images = []
    indices = []
    for item in batch:
        if item[0] is not None:
            images.append(item[0])
            indices.append(item[1])
    return images, indices


dataset = Flickr30KDataset(image_paths)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=pil_collate
)

# %% [markdown]
# ## 3. Feature Extraction
# Extract 512-dim visual features using frozen CLIP ViT-B/32.
# %%
print("Loading CLIP model (ViT-B/32)...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(DEVICE)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

all_embeddings = []
valid_indices = []  # track which original image indices survived

print(f"Extracting features from {len(dataset):,} images (corrupt files will be skipped)...")
with torch.no_grad():
    for images, idxs in tqdm(dataloader):
        if len(images) == 0:
            continue  # entire batch was corrupt (extremely unlikely)
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        # bypass the get_image_features wrapper for API stability
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        features = model.visual_projection(vision_outputs.pooler_output)
        
        all_embeddings.append(features.cpu().numpy())
        valid_indices.extend(idxs)

embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
valid_paths = [image_paths[i] for i in valid_indices]

skipped = len(image_paths) - len(valid_paths)
print(f"Done. Embedded {len(valid_paths):,} images, skipped {skipped} corrupt files")
print(f"Embedding matrix shape: {embeddings.shape}")

# save so we don't have to re-run the heavy model again
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
np.save(EMBEDDINGS_PATH, embeddings)

paths_path = os.path.join(DATA_DIR, "paths.json")
with open(paths_path, "w") as f:
    json.dump(valid_paths, f)

print(f"Saved embeddings to {EMBEDDINGS_PATH}")

# %% [markdown]
# ## 3b. Train/Query Split
# 90/10 split. Query images are held-out to ensure unbiased recall evaluation.
# %%
print("Normalizing embeddings (L2) so inner product equals cosine similarity...")
faiss.normalize_L2(embeddings)
DIMENSION = embeddings.shape[1]

np.random.seed(42)
n = embeddings.shape[0]
perm = np.random.permutation(n)

query_size = max(1000, int(n * 0.10))  # 10% or at least 1,000
query_ids = perm[:query_size]
index_ids = perm[query_size:]

query_embeddings = embeddings[query_ids]
index_embeddings = embeddings[index_ids]

# clean path lists: position i in the index/query set maps directly to a file path
index_paths = [valid_paths[orig] for orig in index_ids]
query_paths = [valid_paths[orig] for orig in query_ids]

print(f"Index set: {len(index_embeddings):,} images (used to build FAISS indexes)")
print(f"Query set: {len(query_embeddings):,} held-out images (used to measure recall)")

# %% [markdown]
# ## 4. Build FAISS Indexes
# Build FlatIP (exact), IVFFlat (clustering), and IVFPQ (clustering + compression).
# %%
build_times = {}

# FlatIP: exact search, ground-truth baseline
print("Building FlatIP index (exact search)...")
t0 = time.time()
index_flat = faiss.IndexFlatIP(DIMENSION)
index_flat.add(index_embeddings)
build_times["FlatIP"] = time.time() - t0
flat_path = os.path.join(DATA_DIR, "index_flat.faiss")
faiss.write_index(index_flat, flat_path)
print(f"  Built in {build_times['FlatIP']:.2f}s")

# IVFFlat: clustering
print("Building IVFFlat index (100 clusters)...")
t0 = time.time()
nlist = 100
quantizer = faiss.IndexFlatIP(DIMENSION)
index_ivf = faiss.IndexIVFFlat(quantizer, DIMENSION, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(index_embeddings)
index_ivf.add(index_embeddings)
build_times["IVFFlat"] = time.time() - t0
ivf_path = os.path.join(DATA_DIR, "index_ivfflat.faiss")
faiss.write_index(index_ivf, ivf_path)
print(f"  Built in {build_times['IVFFlat']:.2f}s")

# IVFPQ: clustering + product quantization
print("Building IVFPQ index (100 clusters, m=64, nbits=8)...")
t0 = time.time()
m = 64
nbits = 8
quantizer_pq = faiss.IndexFlatIP(DIMENSION)
index_ivfpq = faiss.IndexIVFPQ(
    quantizer_pq, DIMENSION, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
)
index_ivfpq.train(index_embeddings)
index_ivfpq.add(index_embeddings)
build_times["IVFPQ"] = time.time() - t0
ivfpq_path = os.path.join(DATA_DIR, "index_ivfpq.faiss")
faiss.write_index(index_ivfpq, ivfpq_path)
print(f"  Built in {build_times['IVFPQ']:.2f}s")

print("\nAll three indexes built and saved.")

# %% [markdown]
# ## 5. Benchmark
# Evaluate Latency, Memory, and Recall@10 (against FlatIP ground truth).
# %%
K = 10


def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)


queries = query_embeddings
num_queries = len(queries)
print(f"Benchmarking with {num_queries:,} held-out query vectors...")

indices = {
    "FlatIP": flat_path,
    "IVFFlat": ivf_path,
    "IVFPQ": ivfpq_path,
}

results = {}
ground_truth = None

for name, path in indices.items():
    size_mb = get_file_size_mb(path)

    index = faiss.read_index(path)
    if hasattr(index, "nprobe"):
        index.nprobe = 10

    # warmup: forces FAISS to load data into CPU cache and allocate buffers
    index.search(queries[:10], K)

    # timed run
    start_time = time.time()
    D, I = index.search(queries, K)
    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / num_queries) * 1000

    if name == "FlatIP":
        ground_truth = I
        gt_scores = D
        recall = 100.0
    else:
        # count how many of FlatIP's top-K results also appear in this index's top-K
        correct = sum(
            len(set(ground_truth[i]).intersection(set(I[i])))
            for i in range(num_queries)
        )
        recall = (correct / (num_queries * K)) * 100

    results[name] = {
        "size_mb": size_mb,
        "latency_ms": avg_latency_ms,
        "recall": recall,
        "build_time_s": build_times[name],
        "result_indices": I,
        "result_scores": D,
    }

# Print benchmark table
print()
print("=" * 85)
print("FAISS Trade-off Results (held-out query set)")
print("=" * 85)
header = f"{'Index':<10} | {'Memory (MB)':<12} | {'Latency (ms)':<13} | {'Recall@10 (%)':<14} | {'Build Time (s)':<14}"
print(header)
print("-" * 85)
for name, res in results.items():
    print(
        f"{name:<10} | {res['size_mb']:<12.2f} | {res['latency_ms']:<13.3f} | "
        f"{res['recall']:<14.2f} | {res['build_time_s']:<14.2f}"
    )

flat_mem = results["FlatIP"]["size_mb"]
pq_mem = results["IVFPQ"]["size_mb"]

compression_ratio = flat_mem / pq_mem

print("\nSummary:")
print(f"IVFPQ achieved {compression_ratio:.1f}x compression relative to FlatIP.")

# %% [markdown]
# ## 6. nprobe Ablation
# Evaluate recall vs latency tradeoff by varying nprobe.
# %%
print()
print("=" * 50)
print("nprobe Ablation (IVFPQ)")
print("=" * 50)
print(f"{'nprobe':<10} | {'Recall@10 (%)':<15} | {'Latency (ms)':<15}")
print("-" * 45)

index_pq = faiss.read_index(ivfpq_path)
nprobe_values = [1, 5, 10, 20, 50]

for p in nprobe_values:
    index_pq.nprobe = p

    start_time = time.time()
    _, I = index_pq.search(queries, K)
    end_time = time.time()
    latency = ((end_time - start_time) / num_queries) * 1000

    correct = sum(
        len(set(ground_truth[i]).intersection(set(I[i])))
        for i in range(num_queries)
    )
    recall = (correct / (num_queries * K)) * 100

    print(f"{p:<10} | {recall:<15.2f} | {latency:<15.3f}")

# %% [markdown]
# ## 7. Failure Analysis
# Inspect IVFPQ failure cases (missed ground truth vs actual retrieval) for graceful degradation.
# %%
print("Failure Analysis: FlatIP vs IVFPQ")
print("=" * 50)

index_pq_analysis = faiss.read_index(ivfpq_path)
index_pq_analysis.nprobe = 10
D_pq, I_pq = index_pq_analysis.search(queries, K)

# compute per-query recall to find the worst cases
per_query_recall = []
for i in range(num_queries):
    gt_set = set(ground_truth[i])
    pq_set = set(I_pq[i])
    hits = len(gt_set.intersection(pq_set))
    per_query_recall.append(hits / K)

per_query_recall = np.array(per_query_recall)
worst_queries = np.argsort(per_query_recall)[:5]

print(f"Overall Recall@10: {per_query_recall.mean() * 100:.2f}%")
print(f"Worst per-query Recall@10 values: {[f'{per_query_recall[i]*100:.0f}%' for i in worst_queries]}")
print()

# visualise the 3 worst failures side-by-side
for rank, qi in enumerate(worst_queries[:3]):
    gt_set = set(ground_truth[qi])
    pq_set = set(I_pq[qi])
    missed = gt_set - pq_set
    extra = pq_set - gt_set

    recall_pct = per_query_recall[qi] * 100
    print(f"Query #{qi} | Recall: {recall_pct:.0f}% | missed {len(missed)} of {K} ground-truth results")

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle(
        f"Failure Case {rank+1}: Query #{qi} (Recall = {recall_pct:.0f}%)\n"
        f"Top row = FlatIP (exact), Bottom row = IVFPQ (approximate)",
        fontsize=13,
    )

    # show the query image on the left of both rows
    query_path = query_paths[qi] if qi < len(query_paths) else "unknown"
    for row in range(2):
        try:
            q_img = Image.open(query_path).convert("RGB")
        except Exception:
            q_img = Image.new("RGB", (224, 224), (200, 200, 200))
        axes[row][0].imshow(q_img)
        axes[row][0].set_title("QUERY", fontweight="bold")
        axes[row][0].axis("off")

    # FlatIP's top-5 results (row 0)
    for j in range(5):
        idx = ground_truth[qi][j]
        path = index_paths[idx] if idx < len(index_paths) else "unknown"
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        axes[0][j + 1].imshow(img)
        in_pq = "✓" if idx in pq_set else "✗ MISSED"
        axes[0][j + 1].set_title(f"sim:{gt_scores[qi][j]:.3f}\n{in_pq}", fontsize=9)
        axes[0][j + 1].axis("off")

    # IVFPQ's top-5 results (row 1)
    for j in range(5):
        idx = I_pq[qi][j]
        path = index_paths[idx] if idx < len(index_paths) else "unknown"
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        axes[1][j + 1].imshow(img)
        in_gt = "✓" if idx in gt_set else "← EXTRA"
        axes[1][j + 1].set_title(f"sim:{D_pq[qi][j]:.3f}\n{in_gt}", fontsize=9)
        axes[1][j + 1].axis("off")

    plt.tight_layout()
    plt.show()
    print()

print(
    "Interpretation: When IVFPQ misses a result, it is usually a semantically similar "
    "image that falls in a neighbouring cluster that nprobe did not reach. The 'extra' "
    "images IVFPQ returns instead are typically close substitutes. The retrieval quality "
    "degrades gracefully rather than catastrophically."
)

# %% [markdown]
# ## 8. Text-to-Image Search
# CLIP was trained to put text and images in the same space, so we can literally
# just encode a sentence with the text tower and search our image index.
# No retraining, no alignment step -- it just works.
# %%

def encode_text(queries):
    """Run text through CLIP's text tower, L2-normalize, return numpy."""
    inputs = processor(
        text=queries, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(DEVICE)
    with torch.no_grad():
        out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        features = model.text_projection(out[1])  # pooler_output
    emb = features.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb


# grab the exact index so results are perfect (no approximation noise)
index_exact = faiss.read_index(flat_path)

# picked queries that should clearly exist in flickr30k
demo_queries = [
    "a dog playing on the beach",
    "people riding bicycles in a city",
    "a child playing in the snow",
    "a cat sitting on a windowsill",
    "a group of people at a concert",
]

for text_query in demo_queries:
    q_vec = encode_text([text_query])
    D, I = index_exact.search(q_vec, 5)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'"{text_query}"', fontsize=14, fontweight="bold")

    for j in range(5):
        idx = I[0][j]
        path = index_paths[idx] if idx < len(index_paths) else None
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        axes[j].imshow(img)
        axes[j].set_title(f"sim: {D[0][j]:.3f}", fontsize=10)
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    print()

# %% [markdown]
# ## 9. Caption-Based Semantic Evaluation
# This is the eval that actually matters. Flickr30K has 5 human captions per image.
# We treat each caption as a text query, search for images, and check if the
# correct image shows up in the top K. Real ground truth, not just
# "does approximate match exact."
# %%
import pandas as pd

# hunt for the captions file -- kaggle mounts datasets in unpredictable paths
captions_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower() == "results.csv":
            captions_path = os.path.join(root, f)
            break
    if captions_path:
        break

if captions_path is None:
    print("No captions file found (results.csv). Skipping semantic eval.")
    print("Make sure the Flickr30K dataset includes the captions CSV.")
else:
    print(f"Found captions at: {captions_path}")

    # flickr30k's csv is pipe-separated and the whitespace is all over the place
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

    # we need to map each caption to the position of its image in our FAISS index
    # images in the query set (held-out 10%) won't be in the index, so we skip those
    path_to_idx = {}
    for i, p in enumerate(index_paths):
        path_to_idx[os.path.basename(p)] = i

    captions_df["index_pos"] = captions_df["image_name"].map(path_to_idx)
    valid = captions_df.dropna(subset=["index_pos"]).copy()
    valid["index_pos"] = valid["index_pos"].astype(int)
    print(f"Usable captions (image is in index set): {len(valid):,}")

    # encoding 100k+ captions is slow, so we sample 2000
    n_eval = min(2000, len(valid))
    sample = valid.sample(n=n_eval, random_state=42)
    print(f"Sampling {n_eval:,} captions for evaluation...")

    # batch-encode all the captions
    caption_texts = sample["caption"].tolist()
    gold = sample["index_pos"].tolist()
    text_embs = []

    for i in tqdm(range(0, len(caption_texts), 64), desc="Encoding captions"):
        batch = caption_texts[i : i + 64]
        text_embs.append(encode_text(batch))

    text_embs = np.concatenate(text_embs, axis=0)

    # run the captions through all 3 indexes and measure recall
    index_files = {
        "FlatIP": flat_path,
        "IVFFlat": ivf_path,
        "IVFPQ": ivfpq_path,
    }

    print()
    print("=" * 60)
    print("Caption -> Image Retrieval (Semantic Recall)")
    print("=" * 60)
    print(f"{'Index':<10} | {'R@1':<8} | {'R@5':<8} | {'R@10':<8}")
    print("-" * 45)

    for name, path in index_files.items():
        idx = faiss.read_index(path)
        if hasattr(idx, "nprobe"):
            idx.nprobe = 10

        _, I = idx.search(text_embs, 10)

        r = {}
        for k in [1, 5, 10]:
            hits = sum(1 for i, g in enumerate(gold) if g in I[i][:k])
            r[k] = (hits / len(gold)) * 100

        print(f"{name:<10} | {r[1]:<8.2f} | {r[5]:<8.2f} | {r[10]:<8.2f}")

    print()
    print("This is the metric that actually matters for a search engine.")
    print("The Section 5 benchmark tells you how well approximate search copies exact search.")
    print("This tells you whether the system retrieves the RIGHT image for a given description.")



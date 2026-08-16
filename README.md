# Visual Search Engine

Image retrieval using CLIP embeddings and FAISS on Flickr30K. The point of this project was to understand the real tradeoffs between exact and approximate nearest-neighbor search -- not just read about them.

I built three different FAISS indexes (exact, clustered, compressed) and benchmarked them against each other on memory, latency, and recall. Then I ran an nprobe ablation to see where the diminishing returns kick in.

The system also does text-to-image search. CLIP puts images and text in the same vector space, so you build the index once from image embeddings and text queries just work -- no extra training or alignment step needed.

## How it works

Every image in Flickr30K goes through a frozen CLIP ViT-B/32 to get a 512-dim embedding vector. L2-normalize everything so that FAISS inner product equals cosine similarity. Split 90/10 into an index set (what we search over) and a held-out query set (what we measure recall with).

Build three indexes on the index set:
- **FlatIP**: brute-force exact search. Perfect recall, but O(n) per query.
- **IVFFlat**: clusters the space into 100 Voronoi cells, only searches the nearest 10 (`nprobe=10`). Same memory, much faster.
- **IVFPQ**: same clustering, but also compresses each vector with Product Quantization (m=64 subspaces, 8 bits each). Huge memory savings, some recall loss.

For text search: encode the query with CLIP's text tower, normalize, search the same image index. No extra alignment needed.

## Evaluation

Two separate evaluations:

**Index tradeoffs (approximate vs exact):** Run held-out image queries through all three indexes. Measures how much recall the approximate methods lose compared to brute force.

**Semantic retrieval (caption-based):** Each Flickr30K image has 5 human captions. Encode each caption with CLIP's text tower, search the image index, check if the correct image appears in the top K. This is the metric that actually matters -- it tests whether the system retrieves the right image for a given description, not just whether approximate search copies exact search.

## What I found

- **IVFFlat is basically free.** It gets 96.3% recall against exact search with similar latency. Unless your corpus is tiny, there's no reason to use brute force.
- **IVFPQ trades recall for memory.** 21x compression (55.9 MB down to 2.7 MB) but recall drops to 65.5%. Worth it only when the full index doesn't fit in RAM.
- **nprobe has sharp diminishing returns.** Going from nprobe=10 to nprobe=50 on IVFPQ: latency goes up 4.5x but recall barely moves (<1% gain). Classic knob you can tune per-application.
- **When IVFPQ misses, it fails gracefully.** The failure analysis visualization shows that "missed" results are usually semantically similar images from neighboring clusters, not random garbage.

## Structure

```
visual-search-engine/
├── kaggle/
│   └── notebook.py           # runs everything end-to-end with visualizations
├── src/
│   ├── feature_extractor.py  # CLIP embeddings + 90/10 split
│   ├── build_index.py        # builds FlatIP / IVFFlat / IVFPQ indexes
│   ├── evaluate_tradeoffs.py # memory / latency / recall benchmark
│   └── evaluate_semantic.py  # caption-based cross-modal retrieval eval
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Create a new Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. Add the [Flickr30K dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) via "Add Data"
4. First cell: `!pip install faiss-cpu`
5. Second cell: paste contents of `kaggle/notebook.py`
6. Run

The notebook auto-detects the Flickr30K path inside `/kaggle/input/`. If it can't find it, it prints debug info showing what Kaggle actually mounted. The caption-based eval runs automatically if `results.csv` is present in the dataset.

### Local
```bash
pip install -r requirements.txt
FLICKR_DIR=/path/to/flickr30k_images python src/feature_extractor.py
python src/build_index.py
python src/evaluate_tradeoffs.py
python src/evaluate_semantic.py
```


# Visual Search Engine
CLIP-based image retrieval with FAISS. Benchmarks exact vs approximate nearest-neighbor search (FlatIP, IVFFlat, IVFPQ) on Flickr30K.

## Architecture
`Image` → `[CLIP ViT-B/32] (frozen)` → `512-dim visual features` → `[L2 Normalization]` → `[FAISS Index] (FlatIP / IVFFlat / IVFPQ)` → `Top-K Similar Images`

## Results & Evaluation
Evaluated on Flickr30K (28k index, 3k query):

| index | memory | latency | recall@10 |
|-------|--------|---------|-----------|
| FlatIP | 55.87 MB | 0.153 ms | 100.00% |
| IVFFlat | 56.28 MB | 0.148 ms | 96.29% |
| IVFPQ | 2.66 MB | 0.075 ms | 65.49% |

IVFPQ compresses footprint 21x with a ~35% recall drop.
## Project Structure
```text
visual-search-engine/
├── kaggle/
│   └── notebook.py           # self-contained kaggle pipeline
├── src/
│   ├── feature_extractor.py  # extract clip embeddings
│   ├── build_index.py        # train faiss indexes
│   └── evaluate_tradeoffs.py # benchmark memory/latency/recall
├── requirements.txt
└── README.md
```

## How to Run

### Kaggle (recommended)
Upload `kaggle/notebook.py` to a Kaggle notebook and run end-to-end.
Enable GPU: Settings -> Accelerator -> GPU T4 x2
Add the [Flickr30K Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).

### Local
```bash
pip install -r requirements.txt
FLICKR_DIR=/path/to/flickr30k_images python src/feature_extractor.py
python src/build_index.py
python src/evaluate_tradeoffs.py
```

# Visual Search Engine

Image search using CLIP and FAISS on Flickr30K. I wanted to understand the tradeoffs between exact and approximate nearest-neighbor search, so I built three different FAISS indexes and benchmarked them properly.

The system does both image-to-image and text-to-image search. CLIP puts images and text in the same vector space, so you build the index once from image embeddings and text queries just work.

## How it works

Every image goes through CLIP ViT-B/32 (frozen) to get a 512-dim vector. I L2-normalize everything so inner product equals cosine similarity, then index with FAISS.

For text search: encode the query with CLIP's text tower, normalize, search the same image index. No extra alignment needed -- that's the whole point of CLIP's shared space.

## Results

### Semantic retrieval (the metric that actually matters)

Each Flickr30K image has 5 human captions. I embed each caption with CLIP's text tower, search the image index, and check if the caption's own image shows up in the top K. This gives a real Recall with ground truth.

| Index | Recall@1 | Recall@5 | Recall@10 |
|-------|----------|----------|-----------|
| FlatIP  | 22.56% | 42.54% | 52.34% |
| IVFFlat | 20.82% | 39.04% | 47.52% |
| IVFPQ   | 14.36% | 31.58% | 39.50% |

### Index tradeoffs

How well each approximate index copies exact search, plus memory and latency:

| Index | Memory | Latency | Recall@10 vs FlatIP |
|-------|--------|---------|---------------------|
| FlatIP  | 55.87 MB | 0.122 ms | 100% (by definition) |
| IVFFlat | 56.28 MB | 0.151 ms | 96.29% |
| IVFPQ   | 2.66 MB  | 0.075 ms | 65.49% |

IVFPQ compresses memory 21x but loses about 35% recall vs exact search.

### nprobe sweep (IVFPQ)

| nprobe | Recall@10 | Latency |
|--------|-----------|---------|
| 1  | 46.67% | 0.012 ms |
| 5  | 63.48% | 0.043 ms |
| 10 | 65.49% | 0.072 ms |
| 20 | 66.04% | 0.135 ms |
| 50 | 66.19% | 0.344 ms |

Going from 10 to 50 nprobe: latency goes up 5x but recall only improves 0.7pp. Classic diminishing returns.

## What I found

- **IVFFlat is basically free.** 96% recall, similar latency. No reason not to use it over brute force unless your corpus is tiny.
- **IVFPQ is a scale-only optimization.** 21x memory savings, but the recall drop is real. Only worth it if the full index doesn't fit in RAM.
- **"Recall vs FlatIP" isn't enough.** It tells you how well the approximate index copies exact search, not whether you're retrieving good images. The caption-based eval is what actually measures retrieval quality.
- **When IVFPQ misses, it fails gracefully.** The failure analysis shows it returns visually similar images from nearby clusters, not random garbage.

## Structure

```
visual-search-engine/
├── kaggle/
│   └── notebook.py           # runs everything end-to-end
├── src/
│   ├── feature_extractor.py  # CLIP embeddings + 90/10 split
│   ├── build_index.py        # FlatIP / IVFFlat / IVFPQ
│   ├── evaluate_tradeoffs.py # memory / latency / recall benchmark
│   ├── eval_text2image.py    # caption-based semantic eval
│   └── text_search.py        # text -> image search CLI
├── app.py                    # Streamlit demo
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Upload `kaggle/notebook.py` to a Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. Add the [Flickr30K dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
4. Run top-to-bottom

### Local
```bash
pip install -r requirements.txt
FLICKR_DIR=/path/to/flickr30k_images python src/feature_extractor.py
python src/build_index.py
python src/evaluate_tradeoffs.py

# semantic eval (needs the captions csv from Flickr30K)
FLICKR_CAPTIONS=/path/to/results.csv python src/eval_text2image.py

# try a text query
python src/text_search.py "a dog running on the beach" --k 5
```

### Streamlit demo
Once the index is built (`data/` folder exists):
```bash
pip install streamlit
streamlit run app.py
```

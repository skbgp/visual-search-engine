# CLIP-based Visual Search Engine with FAISS

This project uses CLIP ViT-B/32 to search Flickr30K by image or by caption. The image and text encoders produce 512-dimensional vectors in the same space. After L2 normalization, FAISS inner-product search ranks them by cosine similarity.

The pipeline uses a seeded 90/10 split of the image embeddings. It builds three indexes on the larger split.

- **FlatIP** searches every vector and provides the exact ranking.
- **IVFFlat** groups the full vectors into 100 clusters and searches a subset of them.
- **IVFPQ** uses the same cluster count and compresses vectors into 64 eight-bit codes.

## Benchmarks

The image benchmark measures how many of FlatIP's top 10 neighbors each index finds on the held-out images. It also reports the serialized index size and the median batch search time divided by the number of queries. The IVFPQ sweep tests `nprobe` values of 1, 5, 10, 20, and 50. The comparison across indexes uses 10 probes for both IVF indexes.

The caption benchmark samples up to 2,000 Flickr30K captions whose images are in the index. A hit means the caption's paired image appears in the top results. This measures zero-shot caption retrieval.

On a full run with 31,783 images, 28,605 went into the index and 3,178 were held out for image queries. The FAISS measurements below used one OpenMP thread on an Apple M4.

| Index | File size | Image Recall@10 | Caption Recall@10 |
| --- | ---: | ---: | ---: |
| FlatIP | 55.87 MiB | 100.00% | 50.50% |
| IVFFlat | 56.28 MiB | 96.29% | 46.00% |
| IVFPQ | 2.66 MiB | 65.63% | 38.70% |

IVFPQ made the serialized index 21× smaller. Moving from 10 to 50 probes raised image Recall@10 by 0.66 percentage points and took about 2.8× as long in this batch benchmark.

## Run it

Use Python 3.10 or newer and run the commands from the repository root. Set `FLICKR_DIR` to the directory containing the JPG files and `FLICKR_CAPTIONS` to the pipe-delimited `results.csv`. The scripts also check the usual local and Kaggle dataset paths. [Flickr30K download](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

```bash
python -m pip install -r requirements.txt
export FLICKR_DIR=/path/to/flickr30k_images
export FLICKR_CAPTIONS=/path/to/results.csv
python -m src.feature_extractor
python -m src.build_index
python -m src.evaluate_tradeoffs
python -m src.evaluate_semantic
```

The extractor saves embeddings and their image paths in `data/`. Rebuild the FAISS indexes after changing the image set. For a small test of the full pipeline, use at least 300 readable images and their matching captions so that PQ has enough vectors to train.

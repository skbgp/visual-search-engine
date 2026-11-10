"""Measure how often a Flickr30K caption retrieves its matching image."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def find_captions():
    configured = os.environ.get("FLICKR_CAPTIONS")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FLICKR_CAPTIONS does not point to a file: {path}")
        return path

    candidates = [
        DATA_DIR / "results.csv",
        DATA_DIR / "flickr30k_images" / "results.csv",
        Path("/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"),
        Path("/kaggle/input/flickr30k-images/results.csv"),
        Path("/kaggle/input/datasets/hsankesara/flickr-image-dataset/flickr30k_images/results.csv"),
    ]
    image_dir = os.environ.get("FLICKR_DIR")
    if image_dir:
        path = Path(image_dir).expanduser()
        candidates.extend((path / "results.csv", path.parent / "results.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Set FLICKR_CAPTIONS to the Flickr30K results.csv file.")


def sample_captions(captions_path, index_paths):
    path_to_idx = {Path(path).name: i for i, path in enumerate(index_paths)}
    if len(path_to_idx) != len(index_paths):
        raise ValueError("Indexed image names must be unique to match Flickr30K captions.")

    records = []
    skipped = 0
    with captions_path.open(encoding="utf-8-sig") as captions:
        next(captions, None)
        for line in captions:
            fields = [field.strip() for field in line.split("|", maxsplit=2)]
            if len(fields) != 3 or not fields[0] or not fields[2]:
                skipped += 1
                continue
            image_name, _, caption = fields
            if image_name in path_to_idx:
                records.append((caption, path_to_idx[image_name]))

    if not records:
        raise ValueError(f"No usable captions in {captions_path} match the indexed images.")
    frame = pd.DataFrame(records, columns=["caption", "index_pos"])
    sample = frame.sample(n=min(2000, len(frame)), random_state=42)
    print(f"{len(frame):,} usable captions, {skipped:,} malformed rows skipped. Evaluating {len(sample):,}.")
    return sample["caption"].tolist(), sample["index_pos"].to_numpy()


def encode_captions(captions):
    # Keep PyTorch and FAISS in separate processes to avoid an OpenMP conflict on macOS.
    with TemporaryDirectory() as directory:
        inputs = Path(directory) / "captions.json"
        outputs = Path(directory) / "embeddings.npy"
        inputs.write_text(json.dumps(captions), encoding="utf-8")
        worker = Path(__file__).with_name("encode_captions.py")
        subprocess.run([sys.executable, str(worker), str(inputs), str(outputs)], check=True)
        return np.load(outputs)


def evaluate():
    captions_path = find_captions()
    paths_file = DATA_DIR / "index_paths.json"
    if not paths_file.is_file():
        raise FileNotFoundError(f"Missing {paths_file}. Run src.feature_extractor first.")
    index_paths = json.loads(paths_file.read_text(encoding="utf-8"))
    caption_texts, gold = sample_captions(captions_path, index_paths)

    index_files = {
        "FlatIP": DATA_DIR / "index_flat.faiss",
        "IVFFlat": DATA_DIR / "index_ivfflat.faiss",
        "IVFPQ": DATA_DIR / "index_ivfpq.faiss",
    }
    indexes = {}
    for name, path in index_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run src.build_index first.")
        if path.stat().st_mtime_ns < paths_file.stat().st_mtime_ns:
            raise ValueError(f"{path.name} predates index_paths.json. Rebuild the indexes.")
        index = faiss.read_index(str(path))
        if index.ntotal != len(index_paths):
            raise ValueError(f"{path.name} does not match index_paths.json. Rebuild the indexes.")
        indexes[name] = index

    text_embeddings = encode_captions(caption_texts)
    faiss.normalize_L2(text_embeddings)

    print("\nCaption-to-image recall (%)")
    print(f"{'Index':<10} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    for name, index in indexes.items():
        if index.d != text_embeddings.shape[1]:
            raise ValueError(f"{name} has the wrong embedding dimension. Rebuild the indexes.")
        if hasattr(index, "nprobe"):
            index.nprobe = min(10, index.nlist)
        _, ids = index.search(text_embeddings, min(10, index.ntotal))
        matches = ids == gold[:, None]
        recalls = [100 * matches[:, :k].any(axis=1).mean() for k in (1, 5, 10)]
        print(f"{name:<10} " + " ".join(f"{recall:>8.2f}" for recall in recalls))


if __name__ == "__main__":
    evaluate()

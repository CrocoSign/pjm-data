"""Build data/texts.h5 (the Polish input of main.py) from the published segment shards.

One HDF5 dataset per clip key, holding the Polish caption of that clip.

Run: uv run --with datasets python build_texts_h5.py
"""

from pathlib import Path

import datasets
import h5py

OUTPUT = Path("data/texts.h5")

dataset = datasets.load_dataset("croco-corp/pjm-segments", split="train")
OUTPUT.parent.mkdir(exist_ok=True)
with h5py.File(OUTPUT, mode="w") as f:
    for record in dataset:
        f.create_dataset(record["__key__"], data=record["txt"])

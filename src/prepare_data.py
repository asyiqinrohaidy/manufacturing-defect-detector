"""
Splits data/raw/<class>/*.jpg into data/processed/{train,val,test}/<class>/*.jpg
Run: python src/prepare_data.py
"""
import os
import random
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42


def main():
    random.seed(SEED)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes: {classes}")

    for cls in classes:
        images = sorted((RAW_DIR / cls).glob("*.jpg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in split_map.items():
            split_dir = OUT_DIR / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, split_dir / f.name)

        print(f"{cls}: train={len(split_map['train'])} val={len(split_map['val'])} test={len(split_map['test'])}")

    print(f"\nDone. Processed data at: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Download a small, balanced subset of the Chest X-Ray Pneumonia dataset.

We pull from a public HuggingFace mirror of the Kaggle "Chest X-Ray Images
(Pneumonia)" dataset, so no Kaggle account or API key is needed. The images
are written to disk in an ``ImageFolder`` layout::

    data/raw/
        train/NORMAL/*.jpeg     train/PNEUMONIA/*.jpeg
        val/NORMAL/*.jpeg       val/PNEUMONIA/*.jpeg
        test/NORMAL/*.jpeg      test/PNEUMONIA/*.jpeg

A handful of images are also copied into ``data/sample/`` so the Streamlit
"Try It" page has example X-rays to show even without the full dataset.

Run::

    python -m src.download_data            # small subset (fast baseline)
    python -m src.download_data --full     # everything the mirror provides
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config


def _save_split(dataset, split_dir: Path, per_class: int | None) -> dict[str, int]:
    """Write up to ``per_class`` images per label into ``split_dir``.

    Returns a count of images written per class label.
    """
    from collections import defaultdict

    written: dict[str, int] = defaultdict(int)
    for label_name in config.CLASS_NAMES:
        (split_dir / label_name).mkdir(parents=True, exist_ok=True)

    # HuggingFace label ids: the mirror uses 0 -> NORMAL, 1 -> PNEUMONIA.
    for i, example in enumerate(dataset):
        label_id = int(example["label"])
        label_name = config.CLASS_NAMES[label_id]
        if per_class is not None and written[label_name] >= per_class:
            # Stop early once every class has hit the cap.
            if all(written[c] >= per_class for c in config.CLASS_NAMES):
                break
            continue

        image = example["image"].convert("RGB")
        out_path = split_dir / label_name / f"{label_name.lower()}_{written[label_name]:04d}.jpeg"
        image.save(out_path, "JPEG", quality=92)
        written[label_name] += 1

    return dict(written)


def _copy_samples(n_per_class: int = 3) -> None:
    """Copy a few test images into data/sample/ for the app's examples."""
    config.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_dir = config.RAW_DATA_DIR / "test"
    if not test_dir.exists():
        return
    for label_name in config.CLASS_NAMES:
        src = test_dir / label_name
        if not src.exists():
            continue
        for j, img in enumerate(sorted(src.glob("*.jpeg"))[:n_per_class]):
            shutil.copy(img, config.SAMPLE_DATA_DIR / f"{label_name.lower()}_example_{j+1}.jpeg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full dataset instead of the small subset.",
    )
    args = parser.parse_args()

    # Imported lazily so `--help` works even before deps are installed.
    from datasets import load_dataset

    config.ensure_dirs()
    print(f"Loading dataset '{config.HF_DATASET_ID}' from HuggingFace ...")
    ds = load_dataset(config.HF_DATASET_ID)
    print("Available splits:", list(ds.keys()))

    if args.full:
        caps = {"train": None, "val": None, "test": None}
    else:
        caps = {
            "train": config.SUBSET_PER_CLASS_TRAIN,
            "val": config.SUBSET_PER_CLASS_VAL,
            "test": config.SUBSET_PER_CLASS_TEST,
        }

    # Some mirrors only ship train/test; synthesise a val split from train if needed.
    split_map = {"train": "train", "test": "test"}
    split_map["val"] = "validation" if "validation" in ds else ("val" if "val" in ds else "train")

    for target_split, cap in caps.items():
        source_split = split_map[target_split]
        data = ds[source_split].shuffle(seed=config.SEED)
        out_dir = config.RAW_DATA_DIR / target_split
        counts = _save_split(data, out_dir, cap)
        print(f"  {target_split:5s} <- {source_split:10s}: {counts}")

    _copy_samples()
    print("\nDone. Data written to", config.RAW_DATA_DIR)
    print("Sample images copied to", config.SAMPLE_DATA_DIR)


if __name__ == "__main__":
    main()

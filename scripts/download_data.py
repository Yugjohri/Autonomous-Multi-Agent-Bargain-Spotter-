"""
Download the Kaggle datasets used to train the INR pricing models into data/raw/<folder>/.

    uv run --group training python scripts/download_data.py [--raw data/raw] [--only amazon_2025 ...]

Folders that already exist are skipped, so files placed there by hand are never touched.
Kaggle credentials are read by kagglehub from the KAGGLE_USERNAME and KAGGLE_KEY environment
variables or from ~/.kaggle/kaggle.json; nothing is read from this repository.

    amazon_2025      prothomeshmistry/amazon-electronics-and-accessories-2025      MIT
    flipkart_2025    priyankamalavade/flipkart-electronics-product-dataset2025    MIT
    amazon_2026      kulkarniparth09/amazon-india-electronics-dataset-2026        Apache 2.0 (test set)
    flipkart_khanna  priyankkhanna/flipkart-product-dataset-by-priyank-khanna     CC BY 4.0
"""

import argparse
import shutil
import sys
from pathlib import Path

DATASETS = {
    "amazon_2025": "prothomeshmistry/amazon-electronics-and-accessories-2025",
    "flipkart_2025": "priyankamalavade/flipkart-electronics-product-dataset2025",
    "amazon_2026": "kulkarniparth09/amazon-india-electronics-dataset-2026",
    "flipkart_khanna": "priyankkhanna/flipkart-product-dataset-by-priyank-khanna",
}


def download(folder: str, handle: str, raw: Path) -> None:
    import kagglehub

    target = raw / folder
    if target.exists():
        print(f"{folder}: {target} exists, skipped")
        return
    cached = Path(kagglehub.dataset_download(handle))
    target.mkdir(parents=True)
    for path in cached.rglob("*.csv"):
        shutil.copy2(path, target / path.name)
    print(f"{folder}: copied {len(list(target.glob('*.csv')))} CSV files from {handle}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--only", nargs="*", choices=list(DATASETS), help="download only these folders")
    args = parser.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    for folder, handle in DATASETS.items():
        if args.only and folder not in args.only:
            continue
        try:
            download(folder, handle, raw)
        except Exception as exc:  # noqa: BLE001 - report and continue with the other datasets
            print(f"{folder}: download failed ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

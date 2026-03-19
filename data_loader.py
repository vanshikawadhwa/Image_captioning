"""
data_loader.py — Load raw captions and image file paths.
"""

import os
import pickle
from glob import glob

from config import IMAGES_DIR, CAPTIONS_PKL


def load_image_paths() -> list[str]:
    """Return a sorted list of absolute paths to every JPEG in IMAGES_DIR."""
    pattern = os.path.join(IMAGES_DIR, "*.jpg")
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No .jpg images found in: {IMAGES_DIR}\n"
            "Make sure the Flickr8k dataset is under data/Flickr8k_Dataset/Images/"
        )
    print(f"[data_loader] Found {len(paths)} images in {IMAGES_DIR}")
    return paths


def load_captions(pkl_path: str = CAPTIONS_PKL) -> dict[str, list[str]]:
    """
    Parse the captions pickle file and return a dict:
        { filename (str) -> [caption1, caption2, ...] }

    The pickle file is expected to contain lines of the form:
        <filename>#<idx>\\t<caption text>
    """
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    captions: dict[str, list[str]] = {}
    for line in raw:
        try:
            parts = line.split("\t")
            filename = parts[0].split("#")[0]
            caption  = parts[1].strip()
            captions.setdefault(filename, []).append(caption)
        except (IndexError, AttributeError):
            continue  # skip malformed lines

    print(f"[data_loader] Loaded captions for {len(captions)} images")
    return captions


def filter_captions(
    captions: dict[str, list[str]],
    image_paths: list[str],
) -> dict[str, list[str]]:
    """
    Keep only caption entries whose filename exists in image_paths.
    Returns a new dict with the same structure.
    """
    filenames = {os.path.basename(p) for p in image_paths}
    filtered = {fn: caps for fn, caps in captions.items() if fn in filenames}
    removed = len(captions) - len(filtered)
    if removed:
        print(f"[data_loader] Dropped {removed} caption entries (images not on disk)")
    print(f"[data_loader] Retained captions for {len(filtered)} images")
    return filtered

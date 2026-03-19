"""
feature_extractor.py — Extract ResNet50 image features and persist them.
"""

import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

from config import (
    FEATURES_PATH,
    IMAGE_SIZE,
    FEATURE_DIM,
    MAX_IMAGES,
)


# ─── GPU Setup ────────────────────────────────────────────────────────────────

def configure_gpu() -> None:
    """Enable dynamic GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[feature_extractor] GPU memory growth enabled ({len(gpus)} device(s))")
        except RuntimeError as exc:
            print(f"[feature_extractor] GPU setup warning: {exc}")
    else:
        print("[feature_extractor] No GPU detected — running on CPU")


# ─── Model ────────────────────────────────────────────────────────────────────

def build_feature_model() -> Model:
    """
    Build a ResNet50-based feature extractor.
    Outputs the 2048-dimensional avg_pool layer (second-to-last layer).
    """
    base = ResNet50(include_top=True)
    feature_layer = base.layers[-2].output          # avg_pool → (2048,)
    model = Model(inputs=base.input, outputs=feature_layer)
    print("[feature_extractor] ResNet50 feature model ready")
    return model


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def load_and_preprocess_image(img_path: str) -> np.ndarray:
    """
    Read an image from disk, resize it to IMAGE_SIZE, and return a
    (1, H, W, 3) float32 array suitable for ResNet50.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.reshape(1, *IMAGE_SIZE, 3).astype(np.float32)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(
    image_paths: list[str],
    model: Model,
    save_path: str = FEATURES_PATH,
    max_images: int | None = MAX_IMAGES,
) -> dict[str, np.ndarray]:
    """
    Extract and cache ResNet50 features for every image.

    If a cached file already exists at *save_path*, it is loaded directly
    (no re-computation).  Otherwise features are computed, saved, and returned.

    Args:
        image_paths: Absolute paths to JPEG images.
        model:       Feature-extraction model (output dim = FEATURE_DIM).
        save_path:   Where to persist / load the pickle cache.
        max_images:  Optional cap on the number of images processed.

    Returns:
        { filename (str) -> feature_vector (np.ndarray, shape (FEATURE_DIM,)) }
    """
    # ── Load from cache ──────────────────────────────────────────────────────
    if os.path.exists(save_path):
        print(f"[feature_extractor] Loading cached features ← {save_path}")
        with open(save_path, "rb") as fh:
            features: dict[str, np.ndarray] = pickle.load(fh)
        print(f"[feature_extractor] Loaded {len(features):,} feature vectors")
        return features

    # ── Compute from scratch ─────────────────────────────────────────────────
    configure_gpu()
    features = {}
    total = len(image_paths) if max_images is None else min(len(image_paths), max_images)
    print(f"[feature_extractor] Extracting features for {total:,} images …")

    for idx, img_path in enumerate(image_paths):
        if max_images is not None and idx >= max_images:
            print(f"[feature_extractor] Reached MAX_IMAGES={max_images}, stopping.")
            break

        try:
            img_array = load_and_preprocess_image(img_path)
            feat = model.predict(img_array, verbose=0).reshape(FEATURE_DIM)
            filename = os.path.basename(img_path)
            features[filename] = feat
        except Exception as exc:
            print(f"[feature_extractor] Skipping {img_path}: {exc}")

        if (idx + 1) % 100 == 0:
            print(f"[feature_extractor]  … {idx + 1:,} / {total:,} processed")

    # ── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as fh:
        pickle.dump(features, fh)
    print(f"[feature_extractor] Features saved → {save_path} ({len(features):,} vectors)")
    return features

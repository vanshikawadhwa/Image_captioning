"""
train.py — End-to-end training pipeline.

Usage:
    python train.py
"""

import os
import numpy as np

from config import (
    BATCH_SIZE, EPOCHS,
    MODEL_PATH, WEIGHTS_PATH, VOCAB_PATH, FEATURES_PATH,
    ARTIFACTS_DIR,
)
from data_loader import load_image_paths, load_captions, filter_captions
from preprocessor import (
    preprocess_all_captions,
    build_vocabulary,
    save_vocabulary,
    encode_captions,
    compute_max_len,
)
from feature_extractor import build_feature_model, extract_features
from dataset import build_training_data
from model import build_model


def train() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    image_paths = load_image_paths()
    raw_captions = load_captions()
    captions = filter_captions(raw_captions, image_paths)

    # ── 2. Extract / load image features ─────────────────────────────────────
    feature_model  = build_feature_model()
    image_features = extract_features(image_paths, feature_model, save_path=FEATURES_PATH)

    # ── 3. Preprocess captions ────────────────────────────────────────────────
    captions = preprocess_all_captions(captions)

    # ── 4. Build vocabulary ───────────────────────────────────────────────────
    vocab = build_vocabulary(captions)
    save_vocabulary(vocab, VOCAB_PATH)

    # ── 5. Encode captions ────────────────────────────────────────────────────
    encoded = encode_captions(captions, vocab)
    max_len = compute_max_len(encoded)

    # ── 6. Build training arrays ──────────────────────────────────────────────
    X, y_in, y_out = build_training_data(image_features, encoded, len(vocab), max_len)

    # ── 7. Build & train model ────────────────────────────────────────────────
    model = build_model(vocab_size=len(vocab), max_len=max_len)
    model.fit(
        [X, y_in],
        y_out,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # ── 8. Save artefacts ─────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")

    model.save_weights(WEIGHTS_PATH)
    print(f"[train] Weights saved → {WEIGHTS_PATH}")

    print("[train] Training complete ✅")


if __name__ == "__main__":
    train()

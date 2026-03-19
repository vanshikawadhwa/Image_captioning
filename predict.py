"""
predict.py — Generate captions for new images using a trained model.

Usage:
    python predict.py                        # captions 5 random images
    python predict.py --image path/to/img    # caption a specific image
    python predict.py --n 10                 # caption 10 random images
"""

import argparse
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

from config import (
    MODEL_PATH, WEIGHTS_PATH, VOCAB_PATH, FEATURES_PATH,
    IMAGE_SIZE, FEATURE_DIM,
    MAX_CAPTION_LEN, START_TOKEN, END_TOKEN,
)
from feature_extractor import build_feature_model, load_and_preprocess_image
from model import load_model
from preprocessor import load_vocabulary


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """Load model, feature extractor, vocabulary, and its inverse mapping."""
    model         = load_model(MODEL_PATH)
    feat_model    = build_feature_model()
    vocab         = load_vocabulary(VOCAB_PATH)
    inv_vocab     = {idx: word for word, idx in vocab.items()}
    return model, feat_model, vocab, inv_vocab


def generate_caption(
    image_path: str,
    model,
    feat_model,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    max_len: int,
) -> str:
    """
    Generate a caption for the image at *image_path*.

    Returns the predicted caption string (without start/end tokens).
    """
    # Extract feature
    img_array = load_and_preprocess_image(image_path)
    feature   = feat_model.predict(img_array, verbose=0).reshape(1, FEATURE_DIM)

    # Greedy decode
    tokens = [START_TOKEN]
    caption_words: list[str] = []

    for _ in range(MAX_CAPTION_LEN):
        encoded = [[vocab.get(t, vocab.get("<OUT>", 0)) for t in tokens]]
        encoded = pad_sequences(encoded, maxlen=max_len, padding="post", truncating="post")

        pred_idx  = int(np.argmax(model.predict([feature, encoded], verbose=0)))
        next_word = inv_vocab.get(pred_idx, "<OUT>")

        if next_word == END_TOKEN:
            break

        caption_words.append(next_word)
        tokens.append(next_word)

    return " ".join(caption_words)


from visualise import visualize_prediction


# ─── Main ─────────────────────────────────────────────────────────────────────

def predict(image_paths: list[str], n: int = 5) -> None:
    """
    Run inference and display predictions.

    Args:
        image_paths: Pool of available image paths to sample from.
        n:           Number of random images to caption (ignored when --image is used).
    """
    model, feat_model, vocab, inv_vocab = load_artifacts()

    # Derive max_len from vocab (saved during training alongside vocab.npy)
    # We load the saved features to determine max_len dynamically if needed.
    # A simple heuristic: use the vocab's embedded MAX_LEN key, or fall back to 40.
    max_len = vocab.get("__MAX_LEN__", 40)

    sample = random.sample(image_paths, min(n, len(image_paths)))
    for img_path in sample:
        caption = generate_caption(img_path, model, feat_model, vocab, inv_vocab, max_len)
        print(f"Image : {os.path.basename(img_path)}")
        print(f"Caption: {caption}\n")
        visualize_prediction(img_path, caption)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_image_paths

    parser = argparse.ArgumentParser(description="Image Captioning — Inference")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to caption.",
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of random images to caption (default: 5).",
    )
    args = parser.parse_args()

    model, feat_model, vocab, inv_vocab = load_artifacts()
    max_len = vocab.get("__MAX_LEN__", 40)

    if args.image:
        # Caption a specific image
        caption = generate_caption(
            args.image, model, feat_model, vocab, inv_vocab, max_len,
        )
        print(f"Caption: {caption}")
        visualize_prediction(args.image, caption)
    else:
        # Caption N random images from the dataset
        all_paths = load_image_paths()
        predict(all_paths, n=args.n)

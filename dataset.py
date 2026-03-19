"""
dataset.py — Build training arrays (X, y_in, y_out) from features + captions.
"""

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from config import OOV_TOKEN


def build_training_data(
    image_features: dict[str, np.ndarray],
    encoded_captions: dict[str, list[list[int]]],
    vocab_size: int,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand every caption into (feature, partial_seq) -> next_word triplets.

    For each caption of length L this produces L-1 training samples using
    teacher-forcing:
      X     — image feature vector                   shape (N, 2048)
      y_in  — left-padded partial caption sequence   shape (N, max_len)
      y_out — one-hot next word                       shape (N, vocab_size)

    Skips any image whose features are not present in *image_features*.

    Returns:
        (X, y_in, y_out) as float64 arrays.
    """
    X_list:    list[np.ndarray] = []
    y_in_list: list[np.ndarray] = []
    y_out_list: list[np.ndarray] = []

    skipped_images = 0

    for filename, seqs in encoded_captions.items():
        if filename not in image_features:
            skipped_images += 1
            continue

        feat = image_features[filename]

        for seq in seqs:
            for i in range(1, len(seq)):
                in_seq  = pad_sequences(
                    [seq[:i]],
                    maxlen=max_len,
                    padding="post",
                    truncating="post",
                )[0]
                out_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]

                X_list.append(feat)
                y_in_list.append(in_seq)
                y_out_list.append(out_seq)

    if skipped_images:
        print(f"[dataset] Skipped {skipped_images} images (no features found)")

    X     = np.array(X_list,     dtype="float64")
    y_in  = np.array(y_in_list,  dtype="float64")
    y_out = np.array(y_out_list, dtype="float64")

    print(
        f"[dataset] Training samples: {len(X):,} | "
        f"X={X.shape}  y_in={y_in.shape}  y_out={y_out.shape}"
    )
    return X, y_in, y_out

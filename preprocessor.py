"""
preprocessor.py — Text preprocessing, vocabulary building, and caption encoding.
"""

import numpy as np

from config import VOCAB_THRESHOLD, OOV_TOKEN, START_TOKEN, END_TOKEN


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def preprocess_caption(text: str) -> str:
    """
    Lowercase a caption and wrap it with start/end tokens.
    Example: "A dog runs." -> "startofseq a dog runs. endofseq"
    """
    text = text.lower().strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"


def preprocess_all_captions(
    captions: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Apply preprocess_caption to every caption in the dict (in-place copy)."""
    return {
        filename: [preprocess_caption(cap) for cap in caps]
        for filename, caps in captions.items()
    }


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocabulary(
    captions: dict[str, list[str]],
    threshold: int = VOCAB_THRESHOLD,
    oov_token: str = OOV_TOKEN,
) -> dict[str, int]:
    """
    Count every word across all captions, keep words with count > threshold,
    and return a word->index mapping.  The OOV token is appended at the end.

    Args:
        captions:  Dict of { filename -> [caption, ...] } (already preprocessed).
        threshold: Words must appear more than this many times to be kept.
                   Default -1 keeps every word.
        oov_token: Token added for unknown words at inference time.

    Returns:
        vocab: { word: int_index }  (1-indexed; 0 is implicitly padding)
    """
    word_counts: dict[str, int] = {}
    for caps in captions.values():
        for cap in caps:
            for word in cap.split():
                word_counts[word] = word_counts.get(word, 0) + 1

    vocab: dict[str, int] = {}
    idx = 1
    for word, count in word_counts.items():
        if count > threshold:
            vocab[word] = idx
            idx += 1

    vocab[oov_token] = len(vocab)  # append OOV at the end
    print(f"[preprocessor] Vocabulary size: {len(vocab):,} words")
    return vocab


def save_vocabulary(vocab: dict[str, int], path: str) -> None:
    """Save vocab dict as a .npy file."""
    np.save(path, vocab)
    print(f"[preprocessor] Vocabulary saved → {path}")


def load_vocabulary(path: str) -> dict[str, int]:
    """Load vocab dict from a .npy file."""
    vocab = np.load(path, allow_pickle=True).item()
    print(f"[preprocessor] Vocabulary loaded ({len(vocab):,} words) ← {path}")
    return vocab


# ─── Caption Encoding ─────────────────────────────────────────────────────────

def encode_captions(
    captions: dict[str, list[str]],
    vocab: dict[str, int],
    oov_token: str = OOV_TOKEN,
) -> dict[str, list[list[int]]]:
    """
    Replace every word in every caption with its integer index from vocab.
    Unknown words are mapped to the OOV index.

    Returns:
        { filename -> [[int, ...], ...] }
    """
    oov_idx = vocab[oov_token]
    encoded: dict[str, list[list[int]]] = {}
    for filename, caps in captions.items():
        encoded[filename] = [
            [vocab.get(word, oov_idx) for word in cap.split()]
            for cap in caps
        ]
    return encoded


def compute_max_len(encoded_captions: dict[str, list[list[int]]]) -> int:
    """Return the length of the longest encoded caption sequence."""
    max_len = max(
        len(seq)
        for seqs in encoded_captions.values()
        for seq in seqs
    )
    print(f"[preprocessor] MAX_LEN = {max_len}")
    return max_len

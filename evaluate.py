"""
evaluate.py — Evaluate the trained captioning model using standard NLP metrics.

Metrics computed:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4  (n-gram precision, most common for captioning)
  - METEOR                            (handles synonyms + stemming)
  - ROUGE-L                           (longest common subsequence)

Usage:
    python evaluate.py               # evaluate on 500 random images (default)
    python evaluate.py --n 200       # evaluate on 200 random images
    python evaluate.py --all         # evaluate on the entire dataset (slow)
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import (
    MODEL_PATH, VOCAB_PATH, FEATURES_PATH,
    FEATURE_DIM, MAX_CAPTION_LEN, START_TOKEN, END_TOKEN,
)
from data_loader import load_image_paths, load_captions, filter_captions
from preprocessor import preprocess_all_captions, load_vocabulary
from feature_extractor import build_feature_model, load_and_preprocess_image
from model import load_model

# ── Ensure NLTK data is available ─────────────────────────────────────────────
def download_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt",            "punkt"),
        ("tokenizers/punkt_tab",        "punkt_tab"),
        ("corpora/wordnet",             "wordnet"),
        ("corpora/omw-1.4",             "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[evaluate] Downloading NLTK resource: {pkg}")
            nltk.download(pkg, quiet=True)


# ── Caption generation ────────────────────────────────────────────────────────

def generate_caption(
    image_path: str,
    model,
    feat_model,
    vocab: dict[str, int],
    inv_vocab: dict[int, str],
    max_len: int,
) -> list[str]:
    """
    Generate a caption for one image.
    Returns a list of word tokens (no start/end tokens).
    """
    from keras.preprocessing.sequence import pad_sequences

    img_array = load_and_preprocess_image(image_path)
    feature   = feat_model.predict(img_array, verbose=0).reshape(1, FEATURE_DIM)

    tokens = [START_TOKEN]
    words: list[str] = []
    oov_idx = vocab.get("<OUT>", 0)

    for _ in range(MAX_CAPTION_LEN):
        encoded = [[vocab.get(t, oov_idx) for t in tokens]]
        encoded = pad_sequences(encoded, maxlen=max_len, padding="post", truncating="post")

        pred_idx  = int(np.argmax(model.predict([feature, encoded], verbose=0)))
        next_word = inv_vocab.get(pred_idx, "<OUT>")

        if next_word == END_TOKEN:
            break

        words.append(next_word)
        tokens.append(next_word)

    return words


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence of two token lists."""
    rows, cols = len(a), len(b)
    dp: list[list[int]] = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[rows][cols]


def rouge_l_sentence(hypothesis: list[str], references: list[list[str]]) -> float:
    """
    Compute sentence-level ROUGE-L F1 against multiple references.
    Returns the best F1 across all references.
    """
    if not hypothesis:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        if not ref:
            continue
        lcs = _lcs_length(hypothesis, ref)
        precision = lcs / len(hypothesis)
        recall    = lcs / len(ref)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(image_paths: list[str], n: Optional[int] = 500) -> Optional[dict]:
    """
    Run evaluation over *n* images (or all if n is None).

    Reports BLEU-1/2/3/4, METEOR, and ROUGE-L.
    """
    download_nltk_data()

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("[evaluate] Loading model …")
    model      = load_model(MODEL_PATH)
    feat_model = build_feature_model()
    vocab      = load_vocabulary(VOCAB_PATH)
    inv_vocab  = {idx: word for word, idx in vocab.items()}
    max_len    = vocab.get("__MAX_LEN__", 40)

    # ── Load & preprocess reference captions ──────────────────────────────────
    print("[evaluate] Loading reference captions …")
    raw_captions = load_captions()
    captions     = filter_captions(raw_captions, image_paths)
    captions     = preprocess_all_captions(captions)

    # ── Sample images to evaluate ─────────────────────────────────────────────
    if n is None:
        sample_paths = image_paths
    else:
        sample_paths = random.sample(image_paths, min(n, len(image_paths)))

    print(f"[evaluate] Evaluating on {len(sample_paths):,} images …\n")

    # ── Accumulators for corpus-level BLEU ────────────────────────────────────
    all_references: list = []    # list[list[list[str]]] — reference lists per image
    all_hypotheses: list = []    # list[list[str]]       — one hypothesis per image
    meteor_scores:  list = []
    rouge_l_scores: list = []

    smooth = SmoothingFunction().method1   # avoids 0 for missing n-grams

    for _img_path in tqdm(sample_paths, desc="Generating captions"):
        img_path = str(_img_path)
        filename = os.path.basename(img_path)

        # Skip images with no reference captions
        if filename not in captions:
            continue

        # References: list of tokenised caption lists (strip start/end tokens)
        refs: list[list[str]] = []
        for cap in captions[filename]:
            tokens = cap.split()
            tokens = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
            refs.append(tokens)

        # Generate hypothesis
        hyp = generate_caption(img_path, model, feat_model, vocab, inv_vocab, max_len)

        if not hyp:
            continue

        all_references.append(refs)
        all_hypotheses.append(hyp)

        # Sentence-level METEOR (best across references)
        best_meteor = max(
            meteor_score([word_tokenize(" ".join(ref))], word_tokenize(" ".join(hyp)))
            for ref in refs
        )
        meteor_scores.append(best_meteor)

        # Sentence-level ROUGE-L
        rouge_l_scores.append(rouge_l_sentence(hyp, refs))

    if not all_hypotheses:
        print("[evaluate] No predictions generated — check model and image paths.")
        return

    # ── Corpus BLEU ───────────────────────────────────────────────────────────
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0),      smoothing_function=smooth)
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0),  smoothing_function=smooth)
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    avg_meteor  = float(np.mean(meteor_scores))
    avg_rouge_l = float(np.mean(rouge_l_scores))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Images evaluated : {len(all_hypotheses):,}")
    print(f"  BLEU-1           : {bleu1:.4f}  ({bleu1 * 100:.2f}%)")
    print(f"  BLEU-2           : {bleu2:.4f}  ({bleu2 * 100:.2f}%)")
    print(f"  BLEU-3           : {bleu3:.4f}  ({bleu3 * 100:.2f}%)")
    print(f"  BLEU-4           : {bleu4:.4f}  ({bleu4 * 100:.2f}%)")
    print(f"  METEOR           : {avg_meteor:.4f}  ({avg_meteor * 100:.2f}%)")
    print(f"  ROUGE-L          : {avg_rouge_l:.4f}  ({avg_rouge_l * 100:.2f}%)")
    print("=" * 50)

    return {
        "bleu1":   bleu1,
        "bleu2":   bleu2,
        "bleu3":   bleu3,
        "bleu4":   bleu4,
        "meteor":  avg_meteor,
        "rouge_l": avg_rouge_l,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning — Evaluation")
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of random images to evaluate on (default: 500).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate on the full dataset (overrides --n).",
    )
    args = parser.parse_args()

    all_paths = load_image_paths()
    eval_n: Optional[int] = None if args.all else args.n
    evaluate(all_paths, n=eval_n)

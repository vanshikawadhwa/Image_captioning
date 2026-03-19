"""
config.py — Central configuration for the Image Captioning project.
All paths, hyperparameters, and constants are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR   = os.path.join(BASE_DIR, "artifacts")

IMAGES_DIR      = os.path.join(DATA_DIR, "Flickr8k_Dataset", "Images")
CAPTIONS_PKL    = os.path.join(DATA_DIR, "set_4.pkl")

FEATURES_PATH   = os.path.join(ARTIFACTS_DIR, "images_features.pkl")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.h5")
WEIGHTS_PATH    = os.path.join(ARTIFACTS_DIR, "mine_model_weights.weights.h5")
VOCAB_PATH      = os.path.join(ARTIFACTS_DIR, "vocab.npy")

# ─── Feature Extraction ───────────────────────────────────────────────────────

IMAGE_SIZE      = (224, 224)          # Input size expected by ResNet50
FEATURE_DIM     = 2048                # Output dimension of ResNet50 (avg_pool layer)
MAX_IMAGES      = None                # Set to an int to cap extraction (None = all)

# ─── Vocabulary ───────────────────────────────────────────────────────────────

VOCAB_THRESHOLD = 5                  # Words with count > THRESH are kept (-1 = all)
OOV_TOKEN       = "<OUT>"            # Out-of-vocabulary token

# ─── Model Hyperparameters ────────────────────────────────────────────────────

EMBEDDING_SIZE  = 128
LSTM_UNITS_1    = 256
LSTM_UNITS_2    = 128
LSTM_UNITS_3    = 512
DROPOUT_RATE    = 0.5

# ─── Training ─────────────────────────────────────────────────────────────────

BATCH_SIZE      = 256
EPOCHS          = 50

# ─── Inference ────────────────────────────────────────────────────────────────

MAX_CAPTION_LEN = 25                  # Maximum words to generate during inference
START_TOKEN     = "startofseq"
END_TOKEN       = "endofseq"

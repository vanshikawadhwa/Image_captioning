"""
model.py — Caption model architecture definition.

Architecture overview:
  - Image branch   : Dense(128) → RepeatVector(max_len) → Dropout
  - Language branch: Embedding → LSTM(256) → TimeDistributed Dense(128)
  - Merge          : Concatenate → LSTM(128) → Dropout → LSTM(512) → Dropout
  - Output         : Dense(vocab_size) → Softmax
"""

from keras.models import Model
from keras.layers import (
    Input, Dense, LSTM, Embedding, RepeatVector,
    TimeDistributed, Concatenate, Activation, Dropout,
)

from config import (
    EMBEDDING_SIZE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    LSTM_UNITS_3,
    DROPOUT_RATE,
    FEATURE_DIM,
)


def build_model(vocab_size: int, max_len: int) -> Model:
    """
    Build and compile the image-captioning model.

    Args:
        vocab_size: Number of unique tokens (including OOV).
        max_len:    Maximum caption length (used for sequence shapes).

    Returns:
        Compiled Keras Model.
    """
    # ── Image Branch ─────────────────────────────────────────────────────────
    image_input     = Input(shape=(FEATURE_DIM,), name="image_input")
    image_emb       = Dense(EMBEDDING_SIZE, activation="relu")(image_input)
    image_emb       = RepeatVector(max_len)(image_emb)
    image_emb       = Dropout(DROPOUT_RATE)(image_emb)

    # ── Language Branch ───────────────────────────────────────────────────────
    lang_input      = Input(shape=(max_len,), name="language_input")
    lang_emb        = Embedding(
                          input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          input_length=max_len,
                      )(lang_input)
    lang_emb        = LSTM(LSTM_UNITS_1, return_sequences=True)(lang_emb)
    lang_emb        = TimeDistributed(Dense(EMBEDDING_SIZE))(lang_emb)

    # ── Merge & Decode ────────────────────────────────────────────────────────
    merged          = Concatenate()([image_emb, lang_emb])
    x               = LSTM(LSTM_UNITS_2, return_sequences=True)(merged)
    x               = Dropout(DROPOUT_RATE)(x)
    x               = LSTM(LSTM_UNITS_3, return_sequences=False)(x)
    x               = Dropout(DROPOUT_RATE)(x)

    # ── Output ────────────────────────────────────────────────────────────────
    x               = Dense(vocab_size)(x)
    output          = Activation("softmax")(x)

    model = Model(inputs=[image_input, lang_input], outputs=output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def load_model(model_path: str) -> Model:
    """Load a saved Keras model from *model_path*."""
    from keras.models import load_model as _load
    model = _load(model_path)
    print(f"[model] Loaded model ← {model_path}")
    return model


def load_weights(model: Model, weights_path: str) -> Model:
    """Load weights into an existing model from *weights_path*."""
    model.load_weights(weights_path)
    print(f"[model] Weights loaded ← {weights_path}")
    return model

"""Sentence-embedding encoder + small MLP classifier.

The encoder is a pre-trained sentence transformer (frozen). Texts are mapped
to fixed-length vectors, and a small MLP learns the refuse/don't-refuse
decision boundary in that space. This is the same broad idea as TF-IDF + LR
but the feature representation is semantic rather than lexical, so synonyms
and paraphrases share signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier


@dataclass
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True  # L2-normalize so cosine similarity == dot product
    device: str | None = None  # None lets sentence-transformers pick (CPU or GPU)


@dataclass
class MLPConfig:
    hidden_layer_sizes: tuple[int, ...] = (128,)
    activation: str = "relu"
    alpha: float = 1e-4  # L2 regularization
    learning_rate_init: float = 1e-3
    max_iter: int = 200
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    random_seed: int = 42


class Embedder:
    """Thin wrapper around SentenceTransformer.encode for batch text encoding."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )


def build_mlp(config: MLPConfig | None = None) -> MLPClassifier:
    config = config or MLPConfig()
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        n_iter_no_change=config.n_iter_no_change,
        random_state=config.random_seed,
    )

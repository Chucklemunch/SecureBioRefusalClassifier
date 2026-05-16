"""TF-IDF + Logistic Regression pipeline factory."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class ModelConfig:
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    C: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    random_seed: int = 42


def build_pipeline(config: ModelConfig | None = None) -> Pipeline:
    config = config or ModelConfig()
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, config.ngram_max),
            min_df=config.min_df,
            max_df=config.max_df,
            sublinear_tf=config.sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=config.C,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            random_state=config.random_seed,
        )),
    ])

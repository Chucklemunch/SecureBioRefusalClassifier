"""TF-IDF + LR specific evaluation helpers.

Model-agnostic helpers (classification report, confusion matrix, ranking,
threshold sweep, per-source breakdown, error analysis, boundary cases,
headline summary) live in `shared.evaluate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def top_features(pipe: Pipeline, k: int = 25) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (top_refuse_features, top_safe_features) as (term, coef) pairs."""
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    top_refuse_idx = np.argsort(coefs)[-k:][::-1]
    top_safe_idx = np.argsort(coefs)[:k]

    top_refuse = [(feature_names[i], float(coefs[i])) for i in top_refuse_idx]
    top_safe = [(feature_names[i], float(coefs[i])) for i in top_safe_idx]
    return top_refuse, top_safe


def print_top_features(pipe: Pipeline, k: int = 25) -> None:
    top_refuse, top_safe = top_features(pipe, k=k)
    print(f"Top {k} features pushing toward REFUSE:")
    for term, coef in top_refuse:
        print(f"  {coef:+.3f}    {term}")
    print(f"\nTop {k} features pushing toward DON'T REFUSE:")
    for term, coef in top_safe:
        print(f"  {coef:+.3f}    {term}")


def held_out_source_eval(
    df: pd.DataFrame,
    held_out_source: str,
    build_pipeline_fn,
) -> dict:
    """Train with `held_out_source` removed; evaluate on those held-out rows only.

    Assumes a text-in sklearn Pipeline. The embedding pipeline does its own
    equivalent inline because it operates on embeddings instead of text.
    """
    mask = df["source"] == held_out_source
    if not mask.any():
        raise ValueError(f"No rows with source={held_out_source!r}")

    train_df = df[~mask]
    test_df = df[mask]

    pipe = build_pipeline_fn()
    pipe.fit(train_df["text"].values, train_df["label"].values)

    y_pred = pipe.predict(test_df["text"].values)
    y_true = test_df["label"].values
    accuracy = (y_pred == y_true).mean()

    return {
        "held_out_source": held_out_source,
        "n": len(test_df),
        "label_distribution": test_df["label"].value_counts().to_dict(),
        "accuracy": float(accuracy),
        "fp_rate": float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else None,
        "fn_rate": float((y_pred[y_true == 1] == 0).mean()) if (y_true == 1).any() else None,
    }

"""Train and evaluate the TF-IDF + LR biosecurity refusal classifier.

Usage:
    python -m tfidf_lr_src.train
    python -m tfidf_lr_src.train --held-out-source mmlu/virology
    python -m tfidf_lr_src.train --save-model models/tfidf_lr.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.data import DatasetConfig, build_dataset, summarize
from shared.evaluate import (
    headline_summary,
    per_source_breakdown,
    print_boundary_cases,
    print_classification_report,
    print_confusion_matrix,
    print_error_analysis,
    print_ranking_metrics,
    print_threshold_sweep,
)
from tfidf_lr_src.evaluate import held_out_source_eval, print_top_features
from tfidf_lr_src.model import ModelConfig, build_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train biosecurity refusal classifier.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--dolly-n", type=int, default=800,
                   help="How many Dolly-15k instructions to use as general negatives.")
    p.add_argument("--held-out-source", type=str, default="mmlu/virology",
                   help="Source string to hold out entirely from training for stress eval. "
                        "Pass empty string to skip.")
    p.add_argument("--save-model", type=str, default=None,
                   help="Optional path to save the fitted pipeline (joblib).")
    p.add_argument("--top-k-features", type=int, default=25)
    p.add_argument("--error-analysis-k", type=int, default=10)
    return p.parse_args()


def section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    args = parse_args()

    section("1. Load & assemble dataset")
    data_cfg = DatasetConfig(dolly_sample_size=args.dolly_n, random_seed=args.seed)
    df = build_dataset(data_cfg)
    summarize(df)

    section("2. Train/test split (stratified by label)")
    # np.asarray(..., dtype=object) sidesteps pandas ArrowDtype, which sklearn's
    # indexing helpers don't handle on recent pandas versions.
    texts = np.asarray(df["text"].tolist(), dtype=object)
    labels = np.asarray(df["label"].tolist(), dtype=int)
    sources = np.asarray(df["source"].tolist(), dtype=object)
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        texts, labels, sources,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"Train: {len(X_train)}    Test: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.3f}    "
          f"Test positive rate: {y_test.mean():.3f}")

    section("3. Fit TF-IDF + LR pipeline")
    model_cfg = ModelConfig(random_seed=args.seed)
    pipe = build_pipeline(model_cfg)
    pipe.fit(X_train, y_train)
    vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
    print(f"Vocabulary size: {vocab_size}")

    section("4. Held-out test set evaluation — per-class metrics")
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print_classification_report(y_test, y_pred)
    print()
    print_confusion_matrix(y_test, y_pred)

    test_df = pd.DataFrame({
        "text": X_test, "label": y_test, "pred": y_pred,
        "prob": y_prob, "source": src_test,
    })

    section("5. Ranking quality & calibration")
    print_ranking_metrics(y_test, y_prob)

    section("6. Threshold sweep — production operating-point selection")
    print_threshold_sweep(y_test, y_prob)

    section("7. Per-source breakdown")
    print(per_source_breakdown(test_df).to_string())

    section("8. Failure modes — confidently-wrong predictions")
    print_error_analysis(test_df, k=args.error_analysis_k)

    section("9. Confusing examples — predictions near the decision boundary")
    print_boundary_cases(test_df, k=args.error_analysis_k)

    section("10. Top features (interpretability sanity check)")
    print_top_features(pipe, k=args.top_k_features)

    if args.held_out_source:
        section(f"11. Stress test: hold out '{args.held_out_source}' from training")
        try:
            result = held_out_source_eval(
                df,
                held_out_source=args.held_out_source,
                build_pipeline_fn=lambda: build_pipeline(model_cfg),
            )
            for k, v in result.items():
                print(f"  {k}: {v}")
            print(
                "\n  Interpretation: high FP rate => the model is using the held-out "
                "topic as a refuse cue, not learning the risk concept itself."
            )
        except ValueError as e:
            print(f"  Skipped: {e}")

    section("12. Headline summary")
    headline_summary(y_test, y_pred, y_prob)

    if args.save_model:
        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_path)
        print(f"\nSaved fitted pipeline to {out_path}")


if __name__ == "__main__":
    main()

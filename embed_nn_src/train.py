"""Train and evaluate the embedding + MLP biosecurity refusal classifier.

Usage:
    python -m embed_nn_src.train
    python -m embed_nn_src.train --held-out-source mmlu/virology
    python -m embed_nn_src.train --save-model models/embed_mlp.joblib

Reuses dataset assembly from `shared.data` and model-agnostic evaluation
helpers from `shared.evaluate`.
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
from embed_nn_src.model import Embedder, EmbedderConfig, MLPConfig, build_mlp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train embedding + MLP biosecurity refusal classifier.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--dolly-n", type=int, default=800,
                   help="Dolly-15k instructions to use as general negatives.")
    p.add_argument("--embedder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace sentence-transformers model id.")
    p.add_argument("--hidden", type=int, nargs="+", default=[128],
                   help="MLP hidden layer sizes, e.g. --hidden 128 or --hidden 256 64.")
    p.add_argument("--alpha", type=float, default=1e-4, help="MLP L2 regularization.")
    p.add_argument("--held-out-source", type=str, default="mmlu/virology",
                   help="Source string to hold out from training for stress eval. "
                        "Pass empty string to skip.")
    p.add_argument("--save-model", type=str, default=None,
                   help="Optional joblib path to persist (embedder model name + fitted MLP).")
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

    texts = df["text"].tolist()
    labels = np.asarray(df["label"].tolist(), dtype=int)
    sources = np.asarray(df["source"].tolist(), dtype=object)

    section(f"2. Encode texts with {args.embedder}")
    embedder = Embedder(EmbedderConfig(model_name=args.embedder))
    embeddings = embedder.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}    (dim={embedder.embedding_dim})")

    section("3. Train/test split (stratified by label)")
    # Split indices so we can recover original texts for error analysis.
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    X_train, X_test = embeddings[idx_train], embeddings[idx_test]
    y_train, y_test = labels[idx_train], labels[idx_test]
    src_test = sources[idx_test]
    texts_test = [texts[i] for i in idx_test]
    print(f"Train: {len(idx_train)}    Test: {len(idx_test)}")
    print(f"Train positive rate: {y_train.mean():.3f}    "
          f"Test positive rate: {y_test.mean():.3f}")

    section("4. Fit MLP on embeddings")
    mlp_cfg = MLPConfig(
        hidden_layer_sizes=tuple(args.hidden),
        alpha=args.alpha,
        random_seed=args.seed,
    )
    mlp = build_mlp(mlp_cfg)
    mlp.fit(X_train, y_train)
    print(f"Architecture: input({embedder.embedding_dim}) -> "
          f"{' -> '.join(str(h) for h in mlp_cfg.hidden_layer_sizes)} -> 1")
    print(f"Iterations until convergence/early-stop: {mlp.n_iter_}")
    print(f"Final training loss: {mlp.loss_:.4f}")

    section("5. Held-out test set evaluation — per-class metrics")
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    print_classification_report(y_test, y_pred)
    print()
    print_confusion_matrix(y_test, y_pred)

    test_df = pd.DataFrame({
        "text": texts_test,
        "label": y_test,
        "pred": y_pred,
        "prob": y_prob,
        "source": src_test,
    })

    section("6. Ranking quality & calibration")
    print_ranking_metrics(y_test, y_prob)

    section("7. Threshold sweep — production operating-point selection")
    print_threshold_sweep(y_test, y_prob)

    section("8. Per-source breakdown")
    print(per_source_breakdown(test_df).to_string())

    section("9. Failure modes — confidently-wrong predictions")
    print_error_analysis(test_df, k=args.error_analysis_k)

    section("10. Confusing examples — predictions near the decision boundary")
    print_boundary_cases(test_df, k=args.error_analysis_k)

    if args.held_out_source:
        section(f"11. Stress test: hold out '{args.held_out_source}' from training")
        mask_ho = sources == args.held_out_source
        if not mask_ho.any():
            print(f"  Skipped: no rows with source={args.held_out_source!r}")
        else:
            X_ho_train = embeddings[~mask_ho]
            y_ho_train = labels[~mask_ho]
            X_ho_test = embeddings[mask_ho]
            y_ho_test = labels[mask_ho]

            mlp_ho = build_mlp(mlp_cfg)
            mlp_ho.fit(X_ho_train, y_ho_train)
            y_ho_pred = mlp_ho.predict(X_ho_test)

            acc = float((y_ho_pred == y_ho_test).mean())
            fp_rate = float((y_ho_pred[y_ho_test == 0] == 1).mean()) if (y_ho_test == 0).any() else None
            fn_rate = float((y_ho_pred[y_ho_test == 1] == 0).mean()) if (y_ho_test == 1).any() else None

            print(f"  held_out_source: {args.held_out_source}")
            print(f"  n: {int(mask_ho.sum())}")
            print(f"  label_distribution: {pd.Series(y_ho_test).value_counts().to_dict()}")
            print(f"  accuracy: {acc:.4f}")
            print(f"  fp_rate: {fp_rate}")
            print(f"  fn_rate: {fn_rate}")
            print("\n  Interpretation: high FP rate => the model is using the held-out "
                  "topic as a refuse cue, not learning the risk concept itself.")

    section("12. Headline summary")
    headline_summary(y_test, y_pred, y_prob)

    if args.save_model:
        out_path = Path(args.save_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist the fitted MLP and the embedder model id (not the encoder weights,
        # which can be re-downloaded by name).
        joblib.dump({
            "embedder_model_name": args.embedder,
            "embedder_normalize": True,
            "mlp": mlp,
            "label_names": ["don't refuse", "refuse"],
        }, out_path)
        print(f"\nSaved fitted model to {out_path}")


if __name__ == "__main__":
    main()

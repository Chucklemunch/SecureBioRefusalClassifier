"""Model-agnostic evaluation helpers.

These work on labels and probabilities regardless of which model produced them
(TF-IDF + LR, embedding + MLP, anything else).

Model-specific helpers live next to the model they're tied to:
  - `top_features` (TF-IDF coefficient inspection) -> `tfidf_lr_src.evaluate`
  - `held_out_source_eval` (text-in pipeline) -> `tfidf_lr_src.evaluate`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


CLASS_NAMES = ["don't refuse", "refuse"]


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    print("=== Confusion matrix ===")
    print(pd.DataFrame(
        cm,
        index=[f"actual {n}" for n in CLASS_NAMES],
        columns=[f"pred {n}" for n in CLASS_NAMES],
    ).to_string())


def per_source_breakdown(test_df: pd.DataFrame) -> pd.DataFrame:
    """Test accuracy and mean refuse-probability by source.

    Useful for catching style/topic confounds: if a benign biology source has
    high mean refuse-probability, the model is mistaking topic for risk.
    """
    return (
        test_df.groupby("source")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "accuracy": (g["pred"] == g["label"]).mean(),
            "mean_refuse_prob": g["prob"].mean(),
        }))
        .sort_values("accuracy")
    )


def print_ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Threshold-independent metrics — how well does the model RANK examples?

    A production refusal classifier usually runs at a tuned threshold, not p=0.5.
    These metrics tell you whether the underlying probabilities are well-ordered.
    Accuracy is included for reference at the default p=0.5 threshold.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float((y_pred == y_true).mean())
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    print("=== Ranking & calibration metrics ===")
    print(f"  Accuracy:    {accuracy:.4f}   (at threshold=0.5; threshold-dependent — see sweep below)")
    print(f"  ROC AUC:     {roc_auc:.4f}   (1.0 = perfect ranking, 0.5 = random)")
    print(f"  PR AUC:      {pr_auc:.4f}    (average precision; emphasizes the positive class)")
    print(f"  Brier score: {brier:.4f}    (lower is better; 0 = perfectly calibrated probabilities)")

    return {"accuracy": accuracy, "roc_auc": roc_auc, "pr_auc": pr_auc, "brier_score": brier}


def print_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """Precision/recall/F1 for the REFUSE class at multiple thresholds."""
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        rows.append({
            "threshold": t,
            "precision_refuse": p,
            "recall_refuse": r,
            "f1_refuse": f1,
            "refuse_rate": float(y_pred.mean()),
        })
    df = pd.DataFrame(rows).set_index("threshold")
    print("=== Threshold sweep (REFUSE class) ===")
    print(df.round(4).to_string())
    return df


def print_error_analysis(test_df: pd.DataFrame, k: int = 10, max_text_chars: int = 280) -> None:
    """Show the highest-confidence FPs and FNs — the model's confidently-wrong cases."""
    wrong = test_df[test_df["pred"] != test_df["label"]].copy()
    if wrong.empty:
        print("No misclassifications on test set.")
        return

    wrong["confidence"] = np.where(wrong["label"] == 0, wrong["prob"], 1 - wrong["prob"])
    wrong = wrong.sort_values("confidence", ascending=False)

    print(f"=== Top {k} false positives (predicted REFUSE, actually benign) ===")
    fps = wrong[wrong["label"] == 0].head(k)
    for _, row in fps.iterrows():
        print(f"[p={row['prob']:.3f}] [{row['source']}]")
        print(f"  {row['text'][:max_text_chars]}")
        print()

    print(f"=== Top {k} false negatives (predicted benign, actually REFUSE) ===")
    fns = wrong[wrong["label"] == 1].head(k)
    for _, row in fns.iterrows():
        print(f"[p={row['prob']:.3f}] [{row['source']}]")
        print(f"  {row['text'][:max_text_chars]}")
        print()


def print_boundary_cases(test_df: pd.DataFrame, k: int = 10, max_text_chars: int = 280) -> None:
    """Show examples closest to the decision boundary — the genuinely confusing ones.

    Different from `print_error_analysis`: that surfaces confident mistakes.
    This surfaces low-confidence predictions, right or wrong — items that
    would flip with a small threshold change.
    """
    df = test_df.copy()
    df["distance_to_boundary"] = (df["prob"] - 0.5).abs()
    df = df.sort_values("distance_to_boundary").head(k)

    print(f"=== Top {k} most uncertain predictions (closest to p=0.5) ===")
    for _, row in df.iterrows():
        outcome = "correct" if row["pred"] == row["label"] else "WRONG"
        true_name = CLASS_NAMES[int(row["label"])]
        pred_name = CLASS_NAMES[int(row["pred"])]
        print(f"[p={row['prob']:.3f}] [{outcome}] true={true_name} pred={pred_name} [{row['source']}]")
        print(f"  {row['text'][:max_text_chars]}")
        print()


def headline_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """One-glance summary of the production-relevant metrics."""
    p_refuse, r_refuse, f1_refuse, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p_safe, r_safe, f1_safe, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=0, zero_division=0
    )
    accuracy = float((y_pred == y_true).mean())
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    print("=== Headline metrics (held-out test set) ===")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  ROC AUC:     {roc_auc:.4f}")
    print(f"  PR AUC:      {pr_auc:.4f}")
    print(f"  REFUSE       precision={p_refuse:.3f}  recall={r_refuse:.3f}  f1={f1_refuse:.3f}")
    print(f"  DON'T REFUSE precision={p_safe:.3f}  recall={r_safe:.3f}  f1={f1_safe:.3f}")

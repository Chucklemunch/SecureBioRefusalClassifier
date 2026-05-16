"""Dataset assembly for the biosecurity refusal classifier.

Positives (label=1): WMDP-bio question stems, no answer choices.
Negatives (label=0): a deliberate mixture so we test risk, not topic.
  - Hard negatives: MMLU biology subjects (same domain, exam-style prose).
  - General negatives: Dolly-15k instructions sample (broad coverage so
    the model doesn't collapse onto "anything non-bio is safe").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset

MMLU_BIO_SUBJECTS: tuple[str, ...] = (
    "college_biology",
    "high_school_biology",
    "virology",
    "medical_genetics",
    "anatomy",
)

MMLU_SPLITS: tuple[str, ...] = ("test", "validation", "dev")


@dataclass
class DatasetConfig:
    dolly_sample_size: int = 800
    random_seed: int = 42


def load_wmdp_bio() -> pd.DataFrame:
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    return pd.DataFrame({
        "text": ds["question"],
        "label": 1,
        "source": "wmdp-bio",
    })


def load_mmlu_bio() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subject in MMLU_BIO_SUBJECTS:
        for split in MMLU_SPLITS:
            try:
                d = load_dataset("cais/mmlu", subject, split=split)
            except Exception as exc:
                print(f"  skipped mmlu/{subject}/{split}: {exc}")
                continue
            frames.append(pd.DataFrame({
                "text": d["question"],
                "source": f"mmlu/{subject}",
            }))
    df = pd.concat(frames, ignore_index=True)
    df["label"] = 0
    return df


def load_dolly_sample(n: int, random_seed: int) -> pd.DataFrame:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = pd.DataFrame({"text": ds["instruction"], "source": "dolly-15k"})
    df = df.sample(n=n, random_state=random_seed).reset_index(drop=True)
    df["label"] = 0
    return df


def build_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    config = config or DatasetConfig()

    parts = [
        load_wmdp_bio(),
        load_mmlu_bio(),
        load_dolly_sample(config.dolly_sample_size, config.random_seed),
    ]
    df = pd.concat(parts, ignore_index=True)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df = df.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)

    # Force plain numpy-backed dtypes. The HuggingFace `datasets` library can
    # produce pyarrow-backed extension arrays that break sklearn's indexing.
    return pd.DataFrame({
        "text": df["text"].tolist(),
        "label": df["label"].astype(int).tolist(),
        "source": df["source"].astype(str).tolist(),
    })


def summarize(df: pd.DataFrame) -> None:
    print(f"Total examples: {len(df)}")
    print("\nClass balance:")
    print(df["label"].value_counts().to_string())
    print("\nSource breakdown:")
    print(df.groupby(["label", "source"]).size().to_string())

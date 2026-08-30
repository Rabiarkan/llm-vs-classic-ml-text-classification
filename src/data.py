"""Data downloading, labeling, sampling, and fixed split generation"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (DATA_PROCESSED, SEED, POOL_SIZE, SAMPLE_SIZE, TEST_SIZE)

CHUNK = 50_000
DATASET = "snap/amazon-fine-food-reviews"
MIN_CHARS, MAX_CHARS = 20, 1500

HTML_TAG = r"<[^>]+>"
WHITESPACE = r"\s+"
HTML_ENTITIES = {
    "&quot;": '"', "&#34;": '"', "&amp;": "&", "&#38;": "&",
    "&apos;": "'", "&#39;": "'", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
}

def check_kaggle_auth() -> None:
    missing = [k for k in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing ENV: {missing}\n"
            "GitHub > Settings > Codespaces > Secrets > REbuild Codespace "
        )
    print(f"✓ Kaggle auth: {os.environ['KAGGLE_USERNAME']}")


def download() -> str:
    import kagglehub
    check_kaggle_auth()
    path = kagglehub.dataset_download(DATASET)
    return os.path.join(path, "Reviews.csv")


def clean_text(s: pd.Series) -> pd.Series:
    s = s.str.replace(HTML_TAG, " ", regex=True)
    for entity, char in HTML_ENTITIES.items():
        s = s.str.replace(entity, char, regex=False)
    return s.str.replace(WHITESPACE, " ", regex=True).str.strip()


def to_label(score: int) -> str:
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def build_pool(csv_path: str) -> pd.DataFrame:
    """Build class-balanced pool"""
    per_class = POOL_SIZE // 3
    parts, need = [], {l: per_class for l in ("negative", "neutral", "positive")}

    reader = pd.read_csv(csv_path, usecols=["Score", "Text"], chunksize=CHUNK)
    for i, chunk in enumerate(reader):
        chunk = chunk.dropna(subset=["Text", "Score"])
        chunk["label"] = chunk["Score"].astype(int).map(to_label)
        chunk["text"] = chunk["Text"].str.strip()
        chunk["text"] = clean_text(chunk["Text"])
        chunk = chunk[chunk["text"].str.len().between(MIN_CHARS, MAX_CHARS)]

        for lbl, remaining in need.items():
            if remaining <= 0:
                continue
            avail = chunk[chunk["label"] == lbl]
            if avail.empty:
                continue
            take = avail.sample(min(remaining, len(avail)), random_state=SEED + i)
            parts.append(take[["text", "label"]])
            need[lbl] -= len(take)

        if all(v <= 0 for v in need.values()):
            break

    pool = pd.concat(parts, ignore_index=True)
    before = len(pool)
    pool = pool.drop_duplicates(subset="text")
    print(f"Duplicate text eliminated: {before - len(pool)}")

    return pool.sample(frac=1, random_state=SEED).reset_index(drop=True)


def make_splits(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pool.groupby("label", group_keys=False).head(SAMPLE_SIZE // 3)
    train, test = train_test_split(
        sample, test_size=TEST_SIZE, random_state=SEED, stratify=sample["label"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    pool = build_pool(download())
    pool.to_csv(DATA_PROCESSED / "pool.csv", index=False)

    train, test = make_splits(pool)
    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test.csv", index=False)

    leak = set(train["text"]) & set(test["text"])
    assert not leak, f"LEAKEGE: {len(leak)} joint text"

    print(f"\npool={len(pool)}  train={len(train)}  test={len(test)}")
    print("\ntrain distribution:\n", train["label"].value_counts().to_string())
    print("\ntest distribution:\n", test["label"].value_counts().to_string())
    print("\nexamples:\n", train.iloc[0]["text"][:200])
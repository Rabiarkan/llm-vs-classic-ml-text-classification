"""
Data downloading, labeling, sampling, and fixed split generation
Preprocessing:
  1. Duplication — the same review has been copied to different ProductIDs
  2. Helpfulness anomaly
  3. Summary leakage — not used; only the Text is read
  4. Class imbalance — preserved; not corrected
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (DATA_PROCESSED, SEED, POOL_SIZE, EVAL_SIZE, EVAL_BAL_SIZE, DEV_SIZE)

CHUNK = 100_000
DATASET = "snap/amazon-fine-food-reviews"
MIN_CHARS, MAX_CHARS = 20, 1500
NEEDED = POOL_SIZE + EVAL_SIZE + DEV_SIZE

HTML_TAG = r"<[^>]+>"
WHITESPACE = r"\s+"
HTML_ENTITIES = {
    "&quot;": '"', "&#34;": '"', "&amp;": "&", "&#38;": "&",
    "&apos;": "'", "&#39;": "'", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
}

USECOLS = ["UserId", "Time", "Score", "Text",
           "HelpfulnessNumerator", "HelpfulnessDenominator"]

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


def build_pool(csv_path: str) -> tuple[pd.DataFrame, dict]:
    stats = {"read": 0, "helpfulness_anomaly": 0,
             "length_filtered": 0, "duplicates": 0}
    parts = []

    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=CHUNK):
        stats["read"] += len(chunk)
        chunk = chunk.dropna(subset=["Text", "Score", "UserId"])

        bad = chunk["HelpfulnessNumerator"] > chunk["HelpfulnessDenominator"]
        stats["helpfulness_anomaly"] += int(bad.sum())
        chunk = chunk[~bad]

        chunk["label"] = chunk["Score"].astype(int).map(to_label)
        chunk["text"] = clean_text(chunk["Text"])

        before = len(chunk)
        chunk = chunk[chunk["text"].str.len().between(MIN_CHARS, MAX_CHARS)]
        stats["length_filtered"] += before - len(chunk)

        parts.append(chunk[["UserId", "Time", "text", "label"]])
        if sum(len(p) for p in parts) > NEEDED * 4:  
            break

    df = pd.concat(parts, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["UserId", "Time", "text"], keep="first")
    df = df.drop_duplicates(subset="text", keep="first")   
    stats["duplicates"] = before - len(df)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df[["text", "label"]], stats


def make_splits(df: pd.DataFrame):
    # eval (frozen) / dev (prompt iteration) / train_pool
    # stratify=label, the natural class proportions are preserved in all sets

    rest, eval_nat = train_test_split(
        df, test_size=EVAL_SIZE, random_state=SEED, stratify=df["label"])
    rest, dev_set = train_test_split(
        rest, test_size=DEV_SIZE, random_state=SEED, stratify=rest["label"])

    per = EVAL_BAL_SIZE // 3
    eval_bal = rest.groupby("label", group_keys=False).head(per)
    rest = rest.drop(eval_bal.index)         

    train_pool = rest.head(POOL_SIZE)

    return (train_pool.reset_index(drop=True), dev_set.reset_index(drop=True),
            eval_nat.reset_index(drop=True), eval_bal.reset_index(drop=True))


if __name__ == "__main__":
    df, stats = build_pool(download())

    for k, v in stats.items():
        print(f"{k:24s}: {v:,}")
    print(f"{'remaning':24s}: {len(df):,}")

    print("\n--- natural class balance ---")
    print((df["label"].value_counts(normalize=True) * 100).round(1).to_string())

    train_pool, dev, eval_nat, eval_bal = make_splits(df)

    splits = {
        "train_pool": train_pool,
        "dev": dev,
        "eval_natural": eval_nat,
        "eval_balanced": eval_bal,
    }
    for name, part in splits.items():
        part.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]["text"]) & set(splits[b]["text"])
            assert not overlap, f"Leakage !!! {a}/{b}: {len(overlap)}"
    print("\n✓ OK")

    print()
    for name, part in splits.items():
        print(f"{name:15s} n={len(part):>6,}")

    print("\neval_natural dist:\n", eval_nat["label"].value_counts().to_string())
    print("\neval_balanced dist:\n", eval_bal["label"].value_counts().to_string())
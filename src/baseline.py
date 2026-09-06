# TF-IDF + LogisticRegression / XGBoost baseline model
"""
2 evaluation sets:
  eval_natural  (1000, 78/14/8)     -> “What does this system do in production?”
  eval_balanced (300, 100/100/100)  -> “Can the model distinguish ‘neutral’?”
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED, SEED, LABELS, PREDICTIONS, TRAIN_SIZES, ANALYSIS
from src.tracker import log_run, timed

EVAL_NAMES = ["natural", "balanced"]


def build_pipeline(kind: str) -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000,
                          sublinear_tf=True, stop_words=None)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced", random_state=SEED)
    elif kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                            eval_metric="mlogloss", tree_method="hist", n_jobs=-1)
    else:
        raise ValueError(kind)
    
    return Pipeline([("tfidf", vec), ("clf", clf)])

def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS,
                                   average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        **{f"f1_{l}": round(float(s), 4) for l, s in zip(LABELS, per_class)},
    }

def report(y_true, y_pred, title: str) -> None:
    print(f"\n= {title} =")
    print(classification_report(y_true, y_pred, labels=LABELS,
                                digits=3, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                       index=[f"true_{l}" for l in LABELS],
                       columns=[f"pred_{l}" for l in LABELS]))

def run_trivial(train: pd.DataFrame, evals: dict) -> None:
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    for name, ev in evals.items():
        pred = dummy.predict(ev["text"])
        m = metrics(ev["label"], pred)
        report(ev["label"], pred, f"trivial (majority) @ eval_{name}")
        log_run("trivial_majority", m, provider="local", model="majority",
                eval_set=name, n_train=len(train), n_test=len(ev),
                cost_usd=0.0, notes="floor — accuracy flatters, macro-F1 does not")

def fit_predict(kind: str, train: pd.DataFrame, evals: dict) -> dict:
    fwd = {l: i for i, l in enumerate(LABELS)}
    y = train["label"].map(fwd) if kind == "xgb" else train["label"]

    pipe = build_pipeline(kind)
    with timed() as t_fit:
        pipe.fit(train["text"], y)

    out = {}
    for name, ev in evals.items():
        with timed() as t_pred:
            raw = pipe.predict(ev["text"])
        pred = np.array([LABELS[i] for i in raw]) if kind == "xgb" else np.asarray(raw)
        out[name] = {"pred": pred,
                     "pred_time_s": t_pred["elapsed_s"],
                     "fit_time_s": t_fit["elapsed_s"]}
    return out

def run(kind: str, train: pd.DataFrame, evals: dict, verbose: bool = True) -> dict:
    res = fit_predict(kind, train, evals)
    summary = {}

    for name, ev in evals.items():
        pred = res[name]["pred"]
        m = metrics(ev["label"], pred)
        summary[name] = m

        if verbose:
            report(ev["label"], pred,
                   f"tfidf_{kind} @ eval_{name} (n_train={len(train):,})")
            saved = ev[["text", "label"]].copy()
            saved["pred"] = pred
            saved.to_csv(PREDICTIONS / f"preds_tfidf_{kind}_{name}.csv", index=False)

        log_run(f"tfidf_{kind}", m, provider="local", model=f"tfidf+{kind}",
                eval_set=name, n_train=len(train), n_test=len(ev),
                train_time_s=res[name]["fit_time_s"],
                latency_p50_s=round(res[name]["pred_time_s"] / len(ev), 6),
                cost_usd=0.0, notes=f"baseline, {len(train)} train rows, stratified split")
    return summary

def learning_curve(pool: pd.DataFrame, evals: dict, n_repeats: int = 3) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    rows = []
    for n in TRAIN_SIZES:
        if n > len(pool):
            print(f"skip: n={n:,} > pool {len(pool):,}")
            continue

        for rep in range(n_repeats):
            if n == len(pool):
                subset = pool
            else:
                subset, _ = train_test_split(
                    pool, train_size=n, random_state=SEED + rep,
                    stratify=pool["label"])

            dist = subset["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
            if (dist < 2).any():
                print(f"skip: n={n}, rep={rep} — class insufficient {dist.to_dict()}")
                continue

            for kind in ("logreg", "xgb"):
                res = run(kind, subset, evals, verbose=False)
                for name, m in res.items():
                    rows.append({"n_train": n, "rep": rep, "model": kind,
                                 "eval_set": name, **m,
                                 **{f"n_{k}": int(v) for k, v in dist.items()}})
                print(f"n={n:>6,} rep={rep}  {kind:7s}  "
                      + "  ".join(f"{k}={v['macro_f1']:.4f}" for k, v in res.items()))

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / "learning_curve.csv", index=False)
    return df

if __name__ == "__main__":
    pool = pd.read_csv(DATA_PROCESSED / "train_pool.csv")
    evals = {name: pd.read_csv(DATA_PROCESSED / f"eval_{name}.csv")
             for name in EVAL_NAMES}

    print(f"train_pool={len(pool):,}  "
          + "  ".join(f"eval_{k}={len(v):,}" for k, v in evals.items()))

    print("\n########## TRIVIAL FLOOR ##########")
    run_trivial(pool, evals)

    n_full = min(TRAIN_SIZES[-1], len(pool))
    print(f"\n########## FULL BASELINE (n_train={n_full:,}) ##########")
    from sklearn.model_selection import train_test_split
    if n_full < len(pool):
        full, _ = train_test_split(pool, train_size=n_full, random_state=SEED,
                                   stratify=pool["label"])
    else:
        full = pool

    print("class distr:",
          full["label"].value_counts().reindex(LABELS).to_dict())

    for kind in ("logreg", "xgb"):
        run(kind, full, evals)
"""
    print("\n########## LEARNING CURVE ##########")
    curve = learning_curve(pool, evals)

    print("\n--- macro-F1 avg ---")
    print(curve.pivot_table(index="n_train", columns=["eval_set", "model"],
                            values="macro_f1", aggfunc="mean").round(4).to_string())

    print("\n--- macro-F1 std ---")
    print(curve.pivot_table(index="n_train", columns=["eval_set", "model"],
                            values="macro_f1", aggfunc="std").round(4).to_string())

    print("\n--- neutral F1 avg ---")
    print(curve.pivot_table(index="n_train", columns=["eval_set", "model"],
                            values="f1_neutral", aggfunc="mean").round(4).to_string())
"""
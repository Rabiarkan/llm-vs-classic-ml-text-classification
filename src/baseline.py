"""
TF-IDF + LogisticRegression / XGBoost baseline model
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, SEED, LABELS, PREDICTIONS, TRAIN_SIZES, ANALYSIS
from src.tracker import log_run, timed

EVAL_NAMES = ["natural", "balanced"]
N_SEEDS = 3
XGB_MAX_N = 10_000

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

def sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(pool):
        return pool
    subset, _ = train_test_split(pool, train_size=n, random_state=seed,
                                 stratify=pool["label"])
    return subset

def run_trivial(train: pd.DataFrame, evals: dict) -> None:
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    for name, ev in evals.items():
        pred = dummy.predict(ev["text"])
        m = metrics(ev["label"], pred)
        report(ev["label"], pred, f"trivial (majority) @ eval_{name}")
        log_run("trivial_majority", m, provider="local", model="majority",
                eval_set=name, n_train=len(train), n_test=len(ev), n_seeds=1,
                cost_usd=0.0, notes="floor — accuracy flatters, macro-F1 does not")

def fit_predict(kind: str, train: pd.DataFrame, evals: dict) -> dict:
    fwd = {l: i for i, l in enumerate(LABELS)}
    y = train["label"].map(fwd) if kind == "xgb" else train["label"]

    pipe = build_pipeline(kind)
    fit_kw = {}
    if kind == "xgb":
        fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", y)
    
    with timed() as t_fit:
        pipe.fit(train["text"], y, **fit_kw)

    out = {}
    for name, ev in evals.items():
        with timed() as t_pred:
            raw = pipe.predict(ev["text"])
        pred = np.array([LABELS[i] for i in raw]) if kind == "xgb" else np.asarray(raw)
        out[name] = {"pred": pred, "fit_time_s": t_fit["elapsed_s"],
                     "pred_time_s": t_pred["elapsed_s"]}
    return out

def run(kind: str, train: pd.DataFrame, evals: dict, seed: int,
        verbose: bool = True, save_preds: bool = False) -> dict:
    res = fit_predict(kind, train, evals)
    summary = {}

    for name, ev in evals.items():
        pred = res[name]["pred"]
        m = metrics(ev["label"], pred)
        summary[name] = m

        if verbose:
            report(ev["label"], pred,
                   f"tfidf_{kind} @ eval_{name} (n_train={len(train):,}, seed={seed})")
        if save_preds:
            saved = ev[["text", "label"]].copy()
            saved["pred"] = pred
            saved.to_csv(PREDICTIONS / f"preds_tfidf_{kind}_{name}.csv", index=False)

        log_run(f"tfidf_{kind}", m, provider="local", model=f"tfidf+{kind}",
                eval_set=name, n_train=len(train), n_test=len(ev), seed=seed,
                train_time_s=res[name]["fit_time_s"],
                latency_p50_s=round(res[name]["pred_time_s"] / len(ev), 6),
                cost_usd=0.0, notes=f"baseline, {len(train)} train rows, stratified, "
                      f"class-weighted, seed={seed}")
    return summary

def run_seeds(kind: str, pool: pd.DataFrame, n: int, evals: dict,
              n_seeds: int = N_SEEDS, verbose: bool = False) -> pd.DataFrame:
    rows = []
    for rep in range(n_seeds):
        seed = SEED + rep
        subset = sample(pool, n, seed)
        dist = subset["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
        if (dist < 2).any():
            print(f"  skipped: n={n}, seed={seed} — class insufficient {dist.to_dict()}")
            continue
 
        res = run(kind, subset, evals, seed,
                  verbose=verbose and rep == 0,
                  save_preds=(rep == 0 and n == max(TRAIN_SIZES)))
        for name, m in res.items():
            rows.append({"n_train": n, "seed": seed, "model": kind,
                         "eval_set": name, **m,
                         **{f"n_{k}": int(v) for k, v in dist.items()}})
    return pd.DataFrame(rows)

def summarise(df: pd.DataFrame, value: str = "macro_f1") -> pd.DataFrame:
    return (df.groupby(["n_train", "model", "eval_set"])[value]
              .agg(["mean", "std", "count"]).round(4).reset_index())

if __name__ == "__main__":
    pool = pd.read_csv(DATA_PROCESSED / "train_pool.csv")
    evals = {n: pd.read_csv(DATA_PROCESSED / f"eval_{n}.csv") for n in EVAL_NAMES}
    print(f"train_pool={len(pool):,}  "
          + "  ".join(f"eval_{k}={len(v):,}" for k, v in evals.items()))
 
    print("\n########## TRIVIAL FLOOR ##########")
    run_trivial(pool, evals)
 
    print(f"\n########## LEARNING CURVE ({N_SEEDS} seeds) ##########")
    frames = []
    for n in TRAIN_SIZES:
        if n > len(pool):
            print(f"skipped: n={n:,} > pool {len(pool):,}")
            continue
        for kind in ("logreg", "xgb"):
            if kind == "xgb" and n > XGB_MAX_N:
                print(f"skipped: xgb @ n={n:,} (duration; this end of the curve is fitted using LogReg)")
                continue
            df = run_seeds(kind, pool, n, evals,
                           verbose=(n == max(TRAIN_SIZES) and kind == "logreg"))
            frames.append(df)
            s = summarise(df)
            for _, r in s.iterrows():
                print(f"n={n:>6,}  {kind:7s} {r['eval_set']:9s} "
                      f"macro_f1={r['mean']:.4f} ± {r['std']:.4f}")
 
    curve = pd.concat(frames, ignore_index=True)
    curve.to_csv(ANALYSIS / "learning_curve.csv", index=False)
 
    print("\n--- macro-F1 (mean ± std, 3 seeds) ---")
    print(summarise(curve).to_string(index=False))
 
    print("\n--- neutral F1 (mean, 3 seeds) ---")
    print(summarise(curve, "f1_neutral").to_string(index=False))
 
    print("\n########## README MAIN TABLE (largest n, mean of seeds) ##########")
    biggest = curve[curve["n_train"] == curve["n_train"].max()]
    for kind in curve["model"].unique():
        for ev in EVAL_NAMES:
            g = curve[(curve["model"] == kind) & (curve["eval_set"] == ev)]
            g = g[g["n_train"] == g["n_train"].max()]
            if g.empty:
                continue
            print(f"tfidf+{kind:6s} @ {ev:9s} n={g['n_train'].iloc[0]:>6,}  "
                  f"macro_f1={g['macro_f1'].mean():.4f} ± {g['macro_f1'].std(ddof=1):.4f}  "
                  f"neutral_f1={g['f1_neutral'].mean():.4f}")
 
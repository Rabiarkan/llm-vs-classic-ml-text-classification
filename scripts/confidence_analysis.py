"""Adım 5 — confidence filter ve human-in-the-loop design.

offline: It is calculated from the existing preds_*.csv files.

Measured variables:
  separation (AUROC): Are high-confidence predictions more accurate?
  calibration (ECE) : Does “0.85, I'm sure” really mean an 85% accuracy rate?
  scope/quality     : What do we gain and what do we lose when we raise the threshold?

"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)

from src.config import LABELS, PREDICTIONS, ANALYSIS

THRESHOLDS = [0.0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_SUBSET = 10          
MIN_PRED_PER_CLASS = 5 # Class F1 calculated using 1-2 predictions is meaningless:
# 120b @ natural, thr=0.95 made a single neutral prediction
# and, knowing the correct answer, produced a macro-F1 of 1.0000

# Assumption regarding the cost of human review
REVIEW_MIN = 0.5                              # minutes per example
HOURLY_USD = 25.0                             # hourly rate for a labelers
REVIEW_USD = REVIEW_MIN / 60 * HOURLY_USD     # ≈ $0.208 per example


FNAME = re.compile(r"preds_(zero_shot|few_shot)_(.+)_(balanced|natural)\.csv")


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: |average confidence − actual hit|, bucket-weighted"""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def analyse(path: Path) -> dict | None:
    m = FNAME.match(path.name)
    if not m:
        return None
    method, model, eval_set = m.groups()
    model = model.replace("openai_", "")

    d = pd.read_csv(path)
    if "confidence" not in d.columns or d["confidence"].isna().all():
        return None
    d = d[d["parse_ok"] & d["confidence"].notna()].copy()
    d["correct"] = d["pred"] == d["true"]

    # --- Threshold scanning ---
    rows = []
    for thr in THRESHOLDS:
        sub = d[d["confidence"] >= thr]
        if len(sub) < MIN_SUBSET:
            continue

        pred_counts = sub["pred"].value_counts().reindex(LABELS).fillna(0).astype(int)
        n_human = len(d) - len(sub)

        rows.append({
            "method": method, "model": model, "eval_set": eval_set,
            "threshold": thr,
            "coverage": round(len(sub) / len(d), 3),
            "precision": round(precision_score(sub["true"], sub["pred"], labels=LABELS,
                                               average="macro", zero_division=0), 4),
            "recall": round(recall_score(sub["true"], sub["pred"], labels=LABELS,
                                         average="macro", zero_division=0), 4),
            "macro_f1": round(f1_score(sub["true"], sub["pred"], labels=LABELS,
                                       average="macro", zero_division=0), 4),
            "accuracy": round(sub["correct"].mean(), 4),
            # Analysis: Is the Rise in Precision Real or Just Caution?
            "n_pred_classes": int((pred_counts > 0).sum()),
            "n_pred_neutral": int(pred_counts["neutral"]),
            "n_auto": len(sub), "n_human": n_human,
            "human_per_1k": round(n_human / len(d) * 1000 * REVIEW_USD, 2),
            "n_human": n_human,
            "valid": bool((pred_counts >= MIN_PRED_PER_CLASS).all()),
        })

    # --- Separation and calibration (threshold-independent) ---
    disc = {
        "method": method, "model": model, "eval_set": eval_set,
        "auroc": round(roc_auc_score(d["correct"], d["confidence"]), 4)
                 if d["correct"].nunique() > 1 else None,
        "ece": round(ece(d["confidence"].to_numpy(), d["correct"].to_numpy()), 4),
        "mean_conf": round(d["confidence"].mean(), 4),
        "accuracy": round(d["correct"].mean(), 4),
        "conf_min": round(d["confidence"].min(), 3),
        "conf_max": round(d["confidence"].max(), 3),
    }
    return {"sweep": pd.DataFrame(rows), "disc": disc}


def best_operating_point(sweep: pd.DataFrame, min_coverage: float = 0.75) -> pd.DataFrame:
    """The best threshold where all three classes are predicted and coverage is preserved.
        “Best precision” is meaningless without a lower bound on the scope: 
        the model could predict a single example and achieve a precision of 1.0.
    """ 
    ok = sweep[sweep["valid"] & (sweep["coverage"] >= min_coverage)]
    idx = ok.groupby(["method", "model", "eval_set"])["precision"].idxmax()
    return ok.loc[idx].sort_values(["eval_set", "precision"], ascending=[True, False])


if __name__ == "__main__":
    sweeps, discs = [], []
    for f in sorted(PREDICTIONS.glob("preds_*.csv")):
        r = analyse(f)          
        if r is not None:
            sweeps.append(r["sweep"])
            discs.append(r["disc"])

    if not sweeps:
        raise SystemExit("preds_{zero_shot,few_shot}_*.csv nt found")

    sweep = pd.concat(sweeps, ignore_index=True)
    disc = pd.DataFrame(discs)
    sweep.to_csv(ANALYSIS / "confidence_sweep.csv", index=False)
    disc.to_csv(ANALYSIS / "confidence_discrimination.csv", index=False)

    print("=" * 78)
    print("1) SEPARATION vs. CALIBRATION  (threshold-independent)")
    print("   AUROC: confidence—can it distinguish between correct and incorrect predictions? (0.5 = random)")
    print("   ECE: How close is the confidence value to the true accuracy? (lower is better)")
    print("=" * 78)
    print(disc.sort_values("auroc", ascending=False).to_string(index=False))

    print("\n" + "=" * 78)
    print("2) THRESHOLD SCREENING")
    print("   valid=False  -> A class is not expected; the line is NOT interpreted")
    print("   If recall decreases as precision increases: it’s not quality, it’s conservatism")

    print("=" * 78)
    cols = ["threshold", "coverage", "precision", "recall", "macro_f1",
            "n_pred_neutral", "n_human", "human_per_1k", "valid"]
    for (meth, mdl, ev), g in sweep.groupby(["method", "model", "eval_set"]):
        print(f"\n--- {meth} | {mdl} | {ev} ---")
        print(g[cols].to_string(index=False))

    print("\n" + "=" * 78)
    print("3) STUDY POINT (Estimated for 3 classes, coverage ≥ 75%)")
    print("=" * 78)
    bop = best_operating_point(sweep)
    print(bop[["method", "model", "eval_set", "threshold", "coverage",
               "precision", "recall", "n_human", "human_per_1k"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("4) COST  (1,000 samples, coverage ≥ 75% at the test point)")
    print("=" * 78)
    print(f"human review: {REVIEW_MIN} min/exmp x ${HOURLY_USD}/hour "
          f"= ${REVIEW_USD:.3f}/exmp")
    print(f"manual (1000 exmp)      : ${1000 * REVIEW_USD:.2f}")
    print(f"oto (LLM, ~$0.05/1k): $0.05")
    print("The cost of an LLM is ~1/4000 of the cost of human labor—the area to optimize is not model selection, but the ratio of samples sent to humans")

    
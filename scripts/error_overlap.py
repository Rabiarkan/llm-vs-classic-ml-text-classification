"""
Do the baseline and LLM make the same mistakes on the same examples?
McNemar test: Is the difference statistically significant?
"""
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from src.config import PREDICTIONS, ANALYSIS, SEED, LABELS

BASELINE = "preds_tfidf_logreg_{eval}.csv"
N_BOOT = 2000
LLM_RUNS = [
    ("zero_shot", "openai_gpt-oss-20b"),
    ("zero_shot", "openai_gpt-oss-120b"),
    ("zero_shot", "claude-haiku-4-5-20251001"),
    ("few_shot", "openai_gpt-oss-20b"),
    ("few_shot", "openai_gpt-oss-120b"),
    ("few_shot", "claude-haiku-4-5-20251001"),
]


def load_pair(eval_set: str, method: str, model: str)  -> pd.DataFrame | None:
    base_path = PREDICTIONS / BASELINE.format(eval=eval_set)
    llm_path = PREDICTIONS / f"preds_{method}_{model}_{eval_set}.csv"
    if not base_path.exists() or not llm_path.exists():
        return None
    
    base = pd.read_csv(base_path)
    llm = pd.read_csv(llm_path)
    llm = llm[llm["parse_ok"]] if "parse_ok" in llm.columns else llm

    m = base.merge(llm[["text", "pred", "confidence"]], on="text",
                   suffixes=("_base", "_llm"))
    if len(m) < 0.9 * len(base):
        print(f"  WARN {method}/{model}/{eval_set}: {len(m)}/{len(base)} matched")
    m["base_ok"] = m["pred_base"] == m["label"]
    m["llm_ok"] = m["pred_llm"] == m["label"]
    return m

def mcnemar(m: pd.DataFrame) -> tuple[int, int, float]:
    b = int((m["base_ok"] & ~m["llm_ok"]).sum())    # only baseline 
    c = int((~m["base_ok"] & m["llm_ok"]).sum())    # only LLM 
    p = binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, float(p)

def paired_bootstrap(y_true, pred_base, pred_llm, n_boot:int = N_BOOT, seed:int = SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(pred_base)
    b = np.asarray(pred_llm)
    n = len(y)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (macro_f1(y[idx], b[idx]) - macro_f1(y[idx], a[idx]) )

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)

def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS,
                          average="macro", zero_division=0))

def analyse(eval_set: str, method: str, model: str) -> tuple[dict, pd.DataFrame] | None:
    m = load_pair(eval_set, method, model)
    if m is None:
        return None
    short = model.replace("openai_", "").replace("-4-5-20251001", "")
    y = m["label"]
    m["pred_oracle"] = np.where(m["base_ok"], m["pred_base"], m["pred_llm"])
 
    both_ok = int((m["base_ok"] & m["llm_ok"]).sum())
    both_bad = int((~m["base_ok"] & ~m["llm_ok"]).sum())
    only_base, only_llm, p_mcnemar = mcnemar(m)
 
    # Jaccard: How much do the error sets overlap?
    n_err_base = int((~m["base_ok"]).sum())
    n_err_llm = int((~m["llm_ok"]).sum())
    union = n_err_base + n_err_llm - both_bad
    jaccard = both_bad / union if union else 0.0

    # --- macro-F1 ---
    f1_base = macro_f1(y, m["pred_base"])
    f1_llm = macro_f1(y, m["pred_llm"])
    f1_oracle = macro_f1(y, m["pred_oracle"])
                         
    # --- accuracy ---
    acc_base = accuracy_score(y, m["pred_base"])
    acc_llm = accuracy_score(y, m["pred_llm"])
    acc_oracle = accuracy_score(y, m["pred_oracle"])

    d_mean, d_lo, d_hi = paired_bootstrap(y, m["pred_base"], m["pred_llm"])

    row = {
        "eval_set": eval_set, "method": method, "model": short, "n": len(m),
        # overlap
        "both_correct": both_ok, "both_wrong": both_bad,
        "only_base": only_base, "only_llm": only_llm,
        "err_jaccard": round(jaccard, 3),
        # meaningfulness
        "mcnemar_p": round(p_mcnemar, 5),
        "f1_delta": round(d_mean, 4),
        "f1_ci_lo": round(d_lo, 4), "f1_ci_hi": round(d_hi, 4),
        "f1_significant": not (d_lo <= 0 <= d_hi),
        # oracle — macro-F1 (primary)
        "base_f1": round(f1_base, 4), "llm_f1": round(f1_llm, 4),
        "oracle_f1": round(f1_oracle, 4),
        "oracle_gain_f1": round(f1_oracle - max(f1_base, f1_llm), 4),
        # oracle — accuracy (secondary)
        "base_acc": round(acc_base, 4), "llm_acc": round(acc_llm, 4),
        "oracle_acc": round(acc_oracle, 4),
        "oracle_gain_acc": round(acc_oracle - max(acc_base, acc_llm), 4),
    }

    return row, m


if __name__ == "__main__":
    rows, samples = [], None

    for eval_set, (method, model) in product(["balanced", "natural"], LLM_RUNS):
        out = analyse(eval_set, method, model)
        if out is None:
            continue
        row, m = out
        rows.append(row)

        # Qualitative examples: Neutral cases detected by the LLM but missed by the baseline
        if samples is None and eval_set == "balanced" and method == "few_shot":
            rescued = m[~m["base_ok"] & m["llm_ok"] & (m["label"] == "neutral")]
            samples = rescued.head(4)[["text", "label", "pred_base",
                                       "pred_llm", "confidence"]]
    if not rows:
        raise SystemExit("preds_* files not found")
 
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / "error_overlap.csv", index=False)
 
    print("=" * 96)
    print("1) ERROR OVERLAP baseline = TF-IDF+LogReg (full pool, class-weighted)")
    print("   err_jaccard 1.0 = They are wrong about these same examples (cascade is useless)")
    print("               0.0 = Completely different examples (complementary)")
    print("=" * 96)
    print(df[["eval_set", "method", "model", "n", "both_wrong",
              "only_base", "only_llm", "err_jaccard"]].to_string(index=False))
 
    print("\n" + "=" * 96)
    print("2) STATISTICAL SIGNIFICANCE")
    print("   mcnemar_p      : Paired ACCURACY test")
    print("   f1_delta [CI]  : macro-F1 difference (LLM vs. baseline), 95% bootstrap interval")
    print("   THE PRIMARY TEST IS BOOTSTRAP — the reported metric is macro-F1")
    print("=" * 96)
    sig = df[["eval_set", "method", "model", "mcnemar_p",
              "f1_delta", "f1_ci_lo", "f1_ci_hi", "f1_significant"]]
    print(sig.to_string(index=False))
 
    n_agree = int((sig["f1_significant"] == (sig["mcnemar_p"] < 0.05)).sum())
    print(f"\n2 test {n_agree}/{len(sig)} yields the same result in the comparison")
 
    print("\n" + "=" * 96)
    print("3a) ORACLE — MACRO-F1")
    print("=" * 96)
    print(df[["eval_set", "method", "model", "base_f1", "llm_f1",
              "oracle_f1", "oracle_gain_f1"]].to_string(index=False))

    print("\n" + "=" * 96)
    print("3b) ORACLE — ACCURACY")
    print("=" * 96)
    print(df[["eval_set", "method", "model", "base_acc", "llm_acc",
              "oracle_acc", "oracle_gain_acc"]].to_string(index=False))

    print("\nAVERAGE ORACLE EARNINGS")
    for ev, g in df.groupby("eval_set"):
        print(f"  {ev:9s} macro-F1 +{g['oracle_gain_f1'].mean():.3f}   "
              f"accuracy +{g['oracle_gain_acc'].mean():.3f}")
        
    if samples is not None and len(samples):
        print("\n" + "=" * 96)
        print("4) NEUTRAL EXAMPLES SAVED BY THE LLM (the baseline got it wrong)")
        print("=" * 96)
        for _, r in samples.iterrows():
            print(f"\n[true {r['label']} | baseline {r['pred_base']} | "
                  f"LLM {r['pred_llm']} conf={r['confidence']}]")
            print(f"  {r['text'][:260]}")
 

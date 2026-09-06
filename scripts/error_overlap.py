"""
Do the baseline and LLM make the same mistakes on the same examples?
McNemar test: Is the difference statistically significant?
"""
import re
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import PREDICTIONS, ANALYSIS

BASELINE = "preds_tfidf_logreg_{eval}.csv"
LLM_RUNS = [
    ("zero_shot", "openai_gpt-oss-20b"),
    ("few_shot", "openai_gpt-oss-20b"),
    ("zero_shot", "claude-haiku-4-5-20251001"),
]


def load_pair(eval_set: str, method: str, model: str):
    base = pd.read_csv(PREDICTIONS / BASELINE.format(eval=eval_set))
    llm_path = PREDICTIONS / f"preds_{method}_{model}_{eval_set}.csv"
    if not llm_path.exists():
        return None
    llm = pd.read_csv(llm_path)
    llm = llm[llm["parse_ok"]]

    m = base.merge(llm[["text", "pred", "confidence"]], on="text",
                   suffixes=("_base", "_llm"))
    if len(m) < 0.9 * len(base):
        print(f"  UYARI: {len(m)}/{len(base)} eşleşti")
    m["base_ok"] = m["pred_base"] == m["label"]
    m["llm_ok"] = m["pred_llm"] == m["label"]
    return m


def mcnemar(m: pd.DataFrame) -> tuple[int, int, float]:
    b = int((m["base_ok"] & ~m["llm_ok"]).sum())    # only baseline 
    c = int((~m["base_ok"] & m["llm_ok"]).sum())    # only LLM 
    p = binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p


if __name__ == "__main__":
    rows, samples = [], []

    for eval_set, (method, model) in product(["balanced", "natural"], LLM_RUNS):
        m = load_pair(eval_set, method, model)
        if m is None:
            continue
        short = model.replace("openai_", "").replace("-4-5-20251001", "")

        both_ok = int((m["base_ok"] & m["llm_ok"]).sum())
        b, c, p = mcnemar(m)
        both_bad = int((~m["base_ok"] & ~m["llm_ok"]).sum())
        n_err_base = int((~m["base_ok"]).sum())
        n_err_llm = int((~m["llm_ok"]).sum())

        # Jaccard: To what extent do the error sets overlap?
        union = n_err_base + n_err_llm - both_bad
        jaccard = both_bad / union if union else 0.0

        rows.append({
            "eval_set": eval_set, "method": method, "model": short,
            "n": len(m),
            "both_correct": both_ok, "both_wrong": both_bad,
            "only_base": b, "only_llm": c,
            "err_jaccard": round(jaccard, 3),
            "mcnemar_p": round(p, 5),
            "verdict": "LLM" if c > b else "baseline",
        })

        # Qualitative: Examples that the LLM saved but the baseline missed
        if eval_set == "balanced" and method == "few_shot":
            rescued = m[~m["base_ok"] & m["llm_ok"] & (m["label"] == "neutral")]
            samples = rescued.head(4)[["text", "label", "pred_base",
                                       "pred_llm", "confidence"]]

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / "error_overlap.csv", index=False)

    print("=" * 78)
    print(" ERROR OVERLAP (baseline = TF-IDF+LogReg @10k)")
    print(" err_jaccard: 1.0 = They are wrong about these same examples (cascade is useless)")
    print("              0.0 = Completely different examples (complementary)")
    print("  mcnemar_p  : paired t-test, p < 0.05 is statistically significant")
    print("=" * 78)
    print(df.to_string(index=False))

    print("\n" + "=" * 78)
    print("UPPER LIMIT: If at least one of the two methods is correct")
    print("=" * 78)
    for _, r in df.iterrows():
        oracle = (r["both_correct"] + r["only_base"] + r["only_llm"]) / r["n"]
        base_acc = (r["both_correct"] + r["only_base"]) / r["n"]
        llm_acc = (r["both_correct"] + r["only_llm"]) / r["n"]
        print(f"{r['eval_set']:9s} {r['method']:10s} {r['model']:14s}  "
              f"baseline {base_acc:.3f}  LLM {llm_acc:.3f}  ORACLE {oracle:.3f}"
              f"  (+{oracle - max(base_acc, llm_acc):.3f})")

    if len(samples):
        print("\n" + "=" * 78)
        print("NEUTRAL EXAMPLES SAVED BY THE LLM (the baseline got it wrong)")
        print("=" * 78)
        for _, r in samples.iterrows():
            print(f"\n[real {r['label']} | baseline {r['pred_base']} | "
                  f"LLM {r['pred_llm']} conf={r['confidence']}]")
            print(f"  {r['text'][:260]}")
"""Prompt iteration — ONLY RUN on dev set (n=200)
Usage:
    python scripts/tune_prompt.py                      # zero-shot
    python scripts/tune_prompt.py --fewshot            # few-shot
    python scripts/tune_prompt.py --fewshot --model openai/gpt-oss-20b
"""
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import DATA_PROCESSED, PREDICTIONS, LABELS, LLM_MODEL
from src.llm import get_provider, provider_for, assert_model_available
from src.llm_classify import classify_all
from src.prompts import (SYSTEM, SYSTEM_VERSION, SYSTEM_FEWSHOT, FEWSHOT_VERSION)
from src.tracker import estimate_cost

load_dotenv()

argv = sys.argv[1:]
few = "--fewshot" in argv
model = argv[argv.index("--model") + 1] if "--model" in argv else LLM_MODEL

system = SYSTEM_FEWSHOT if few else SYSTEM
version = FEWSHOT_VERSION if few else SYSTEM_VERSION
tag = "fewshot" if few else "zeroshot"

assert SYSTEM_FEWSHOT.startswith(SYSTEM), "SYSTEM_FEWSHOT base broken"

dev = pd.read_csv(DATA_PROCESSED / "dev.csv")
extra = (len(SYSTEM_FEWSHOT) - len(SYSTEM)) // 4 if few else 0
print(f"{version} @ {model} | dev n={len(dev)} | +{extra} token/call")

p = get_provider(provider_for(model), model)
assert_model_available(p)

res = classify_all(p, dev["text"].tolist(), system=system,
                   desc=f"dev/{tag}",
                   checkpoint=PREDICTIONS / f"partial_dev_{tag}.csv")

if len(res) < len(dev):
    raise SystemExit(f"INCOMPLETE RUN ({len(res)}/{len(dev)}) — isnot logged")

res["true"] = dev["label"].values

n_bad = int((~res["parse_ok"]).sum())
n_ok = len(res) - n_bad
print(f"\nschema validation: {n_ok}/{len(res)} passed"
      + (f" — {n_bad} failed" if n_bad else ""))
if n_bad:
    print(res.loc[~res["parse_ok"], ["parse_error", "raw"]].head(3).to_string())

ok = res[res["parse_ok"]]

print(classification_report(ok["true"], ok["pred"], labels=LABELS,
                            digits=3, zero_division=0))
print(pd.DataFrame(confusion_matrix(ok["true"], ok["pred"], labels=LABELS),
                   index=[f"true_{l}" for l in LABELS],
                   columns=[f"pred_{l}" for l in LABELS]))

macro = f1_score(ok["true"], ok["pred"], labels=LABELS,
                 average="macro", zero_division=0)
per_class = dict(zip(LABELS, f1_score(ok["true"], ok["pred"], labels=LABELS,
                                      average=None, zero_division=0)))
print(f"\nmacro-F1: {macro:.4f}   " +
      "  ".join(f"{k}={v:.4f}" for k, v in per_class.items()))

cost = estimate_cost(int(res["input_tokens"].sum()),
                     int(res["output_tokens"].sum()), model)
print(f"input tokens: {int(res['input_tokens'].sum()):,}  "
      f"cost/1k: ${cost / len(res) * 1000:.4f}")

print("\nconfidence, by class:")
print(ok.groupby("pred")["confidence"].agg(["mean", "count"]).round(3).to_string())

bins = pd.cut(ok["confidence"], [0, .7, .85, .95, 1])
acc = ok.assign(correct=ok["pred"] == ok["true"]).groupby(
    bins, observed=True)["correct"].agg(["mean", "count"]).round(3)
print(acc.to_string())

# --- zero-shot ---
prev = PREDICTIONS / "dev_zeroshot_preds.csv"
if few and prev.exists():
    base = pd.read_csv(prev)
    m = base.merge(ok[["text", "pred"]], on="text", suffixes=("_zs", "_fs"))
    m["true"] = m["text"].map(dict(zip(dev["text"], dev["label"])))
    fixed = ((m["pred_zs"] != m["true"]) & (m["pred_fs"] == m["true"])).sum()
    broke = ((m["pred_zs"] == m["true"]) & (m["pred_fs"] != m["true"])).sum()
    print(f"\nAccording to zero-shot: fixed {fixed}, broken {broke}, total {fixed - broke:+d}")

print("\n--- wrong prediction ---")
wrong = ok[ok["pred"] != ok["true"]]
for _, r in wrong.head(10).iterrows():
    print(f"[{r['true']} → {r['pred']}, conf={r['confidence']}] {r['reason']}")
    print(f"    {r['text'][:170]}\n")

res.to_csv(PREDICTIONS / f"dev_{tag}_preds.csv", index=False)
print(f"records: results/dev_{tag}_preds.csv")
# Prompt iteration — ONLY the dev set (n=200)

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score

from src.config import DATA_PROCESSED, LABELS, PROVIDER, LLM_MODEL
from src.llm import get_provider, assert_model_available
from src.llm_classify import classify_all
from src.prompts import SYSTEM

load_dotenv()

dev = pd.read_csv(DATA_PROCESSED / "dev.csv")

p = get_provider(PROVIDER, LLM_MODEL)
assert_model_available(p)

res = classify_all(p, dev["text"].tolist(), system=SYSTEM, desc="dev")
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
print("macro-F1:", round(f1_score(ok["true"], ok["pred"], labels=LABELS,
                                  average="macro", zero_division=0), 4))

print("\nconfidence distribution:")
print(ok["confidence"].describe().round(3).to_string())
print("\naverage confidence per class:")
print(ok.groupby("pred")["confidence"].agg(["mean", "count"]).round(3).to_string())

print("\n--- incorrect predictions (by the model's reasoning) ---")
wrong = ok[ok["pred"] != ok["true"]]
for _, r in wrong.head(8).iterrows():
    print(f"[{r['true']} → {r['pred']}, conf={r['confidence']}] {r['reason']}")
    print(f"    {r['text'][:180]}\n")
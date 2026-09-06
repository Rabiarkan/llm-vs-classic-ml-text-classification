"""
LLM final evaluation

Usage:
    python scripts/run_llm.py openai/gpt-oss-20b --balanced
    python scripts/run_llm.py openai/gpt-oss-20b --balanced --fewshot
    python scripts/run_llm.py claude-haiku-4-5-20251001 --balanced --natural --fewshot
"""
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import DATA_PROCESSED, RESULTS, LABELS, BENCH_MODELS_FREE, PREDICTIONS
from src.llm import get_provider, provider_for, assert_model_available
from src.llm_classify import classify_all
from src.prompts import SYSTEM, SYSTEM_VERSION, SYSTEM_FEWSHOT, FEWSHOT_VERSION
from src.tracker import estimate_cost, guard_budget, log_run

load_dotenv()

FLAGS = {"--fewshot"}

argv = sys.argv[1:]
few = "--fewshot" in argv
models = [a for a in argv if not a.startswith("--")] or BENCH_MODELS_FREE
only = [a[2:] for a in argv if a.startswith("--") and a not in FLAGS]

EVALS = {n: pd.read_csv(DATA_PROCESSED / f"eval_{n}.csv")
         for n in ["balanced", "natural"]}
if only:
    unknown = set(only) - set(EVALS)
    if unknown:
        raise SystemExit(f"unknown eval set: {unknown} (valid: {set(EVALS)})")
    EVALS = {k: v for k, v in EVALS.items() if k in only}

assert SYSTEM_FEWSHOT.startswith(SYSTEM), "SYSTEM_FEWSHOT broke the base SYSTEM string"

system = SYSTEM_FEWSHOT if few else SYSTEM
version = FEWSHOT_VERSION if few else SYSTEM_VERSION
method = "few_shot" if few else "zero_shot"
extra_tok = (len(SYSTEM_FEWSHOT) - len(SYSTEM)) // 4 if few else 0

print(f"method: {method} ({version})" + (f"  +{extra_tok} token/call" if few else ""))
print(f"models: {models}\neval sets: {list(EVALS)}")


def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {"macro_f1": round(f1_score(y_true, y_pred, labels=LABELS,
                                       average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            **{f"f1_{l}": round(float(s), 4) for l, s in zip(LABELS, per_class)}}


for model in models:
    p = get_provider(provider_for(model), model)
    assert_model_available(p)

    for eval_name, ev in EVALS.items():
        print(f"\n{'=' * 60}\n{method} | {model} @ eval_{eval_name} (n={len(ev)})\n{'=' * 60}")

        # few-shot prompt'u ~3.5x uzun; bütçe tahmini bunu yansıtmalı
        tok_in = 180 + extra_tok
        guard_budget(len(ev), model, provider=p.name)

        if p.name == "anthropic":
            est = estimate_cost(len(ev) * tok_in, len(ev) * 40, model)
            if input(f"~${est:.3f} will be spent. Continue? [y/N] ").strip().lower() != "y":
                print("cancel")
                continue

        safe = model.replace("/", "_")
        stem = f"{method}_{safe}_{eval_name}"

        res = classify_all(p, ev["text"].tolist(), system=system,
                           desc=f"{safe}/{eval_name}",
                           checkpoint=RESULTS / f"partial_{stem}.csv")

        if len(res) < len(ev):
            print(f"INCOMPLETE RUN ({len(res)}/{len(ev)}) — isnot logged. "
                  "Missing data...")
            continue

        res["true"] = ev["label"].values
        n_bad = int((~res["parse_ok"]).sum())
        n_ok = len(res) - n_bad
        print(f"\nschema validation: {n_ok}/{len(res)} passed"
              + (f" — {n_bad} failed" if n_bad else ""))

        res["pred_eval"] = res["pred"].fillna("__parse_error__")

        print(classification_report(res["true"], res["pred_eval"],
                                    labels=LABELS, digits=3, zero_division=0))
        print(pd.DataFrame(
            confusion_matrix(res["true"], res["pred_eval"], labels=LABELS),
            index=[f"true_{l}" for l in LABELS],
            columns=[f"pred_{l}" for l in LABELS]))

        cost = estimate_cost(int(res["input_tokens"].sum()),
                             int(res["output_tokens"].sum()), model)
        log_run(method, metrics(res["true"], res["pred_eval"]),
                provider=p.name, model=model, eval_set=eval_name,
                n_train=0, n_shot=4 if few else 0, n_test=len(res),
                n_parse_failed=n_bad,
                input_tokens=int(res["input_tokens"].sum()),
                output_tokens=int(res["output_tokens"].sum()),
                cost_usd=round(cost, 5),
                cost_per_1k_usd=round(cost / len(res) * 1000, 4),
                latency_p50_s=round(float(res["latency_s"].median()), 3),
                latency_p95_s=round(float(res["latency_s"].quantile(0.95)), 3),
                mean_confidence=round(float(res["confidence"].mean(skipna=True)), 3),
                notes=f"{method}, prompt={version}, temperature=0")

        res.to_csv(PREDICTIONS / f"preds_{stem}.csv", index=False)
        print(f"records: results/predictions/preds_{stem}.csv")
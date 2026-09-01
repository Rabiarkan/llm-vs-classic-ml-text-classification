# Zero-shot final evaluation: each model × each evaluation set, one round.

import sys
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.config import (DATA_PROCESSED, RESULTS, LABELS, PROVIDER,
                        BENCH_MODELS_FREE)
from src.llm import get_provider, assert_model_available
from src.llm_classify import classify_all
from src.prompts import SYSTEM, SYSTEM_VERSION
from src.tracker import estimate_cost, guard_budget, log_run

load_dotenv()

EVALS = {name: pd.read_csv(DATA_PROCESSED / f"eval_{name}.csv")
         for name in ["balanced", "natural"]}   


def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS,
                                   average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        **{f"f1_{l}": round(float(s), 4) for l, s in zip(LABELS, per_class)},
    }


models = sys.argv[1:] or BENCH_MODELS_FREE

for model in models:
    p = get_provider(PROVIDER, model)
    assert_model_available(p)

    for eval_name, ev in EVALS.items():
        print(f"\n{'=' * 60}\n{model} @ eval_{eval_name} (n={len(ev)})\n{'=' * 60}")
        guard_budget(len(ev), model, max_usd=0.50)

        res = classify_all(p, ev["text"].tolist(), system=SYSTEM,
                           desc=f"{model.split('/')[-1]}/{eval_name}")
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
        log_run(
            "zero_shot", metrics(res["true"], res["pred_eval"]),
            provider=p.name, model=model, eval_set=eval_name,
            n_train=0, n_test=len(res), n_parse_failed=n_bad,
            input_tokens=int(res["input_tokens"].sum()),
            output_tokens=int(res["output_tokens"].sum()),
            cost_usd=round(cost, 5),
            cost_per_1k_usd=round(cost / len(res) * 1000, 4),
            latency_p50_s=round(float(res["latency_s"].median()), 3),
            latency_p95_s=round(float(res["latency_s"].quantile(0.95)), 3),
            mean_confidence=round(float(res["confidence"].mean(skipna=True)), 3),
            notes="zero-shot, prompt={SYSTEM_VERSION}, temperature=0",
        )

        safe = model.replace("/", "_")
        res.to_csv(RESULTS / f"preds_zeroshot_{safe}_{eval_name}.csv", index=False)
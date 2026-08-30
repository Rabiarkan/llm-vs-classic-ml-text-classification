""" Measurements: cost calculation, budget protection, time measurement, experiment log"""
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import RUNS_LOG, PRICING


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    p_in, p_out = PRICING[model]
    if p_in is None or p_out is None:
        raise ValueError(f"No price defined for {model} — fill in PRICING")
    return input_tokens / 1_000_000 * p_in + output_tokens / 1_000_000 * p_out


def guard_budget(n_examples: int, model: str, max_usd: float = 0.50) -> float:
    est = estimate_cost(n_examples * 180, n_examples * 30, model)
    if est > max_usd:
        raise RuntimeError(f"Estimated ${est:.3f} > limit ${max_usd:.2f} — cancel")
    print(f"Budget OK: ~${est:.4f} ({n_examples} example, {model})")
    return est


@contextmanager
def timed():
    """with timed() as t: ...   ->   t['elapsed_s']"""
    box = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["elapsed_s"] = round(time.perf_counter() - start, 3)


def log_run(method: str, metrics: dict, **extra) -> dict:
    """Adds experiment log to results/runs.jsonl """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,        # tfidf_logreg | zero_shot | few_shot | few_shot_conf
        **metrics,               # macro_f1, accuracy, precision, recall ...
        **extra,                 # provider, model, n_test, latency_p50_s,
                                 # input_tokens, output_tokens, cost_usd, notes
    }
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
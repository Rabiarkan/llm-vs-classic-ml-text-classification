"""
LLM classification run: training, validation, evaluation.
Measurement guidelines:
  - Latency is measured per request; it does NOT include retry wait times
  - Token counts are taken from the provider’s “usage” field, not estimated
  - Parsing errors are flagged separately from model errors
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from tenacity import (retry, retry_if_not_exception_type, stop_after_attempt,
                      wait_exponential)
from tqdm import tqdm

from src.llm import LLMProvider
from src.prompts import SYSTEM, USER_TEMPLATE
from src.schema import Prediction


JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class QuotaExhausted(Exception):
    """Raised when the provider returns a quota-exhausted error"""
    pass

def _is_daily_quota(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg and any(k in msg for k in ("per day", "tpd", "rpd"))

def parse(raw: str) -> Prediction:
    m = JSON_BLOCK.search(raw or "")
    if not m:
        raise ValueError(f"JSON not found: {(raw or '')[:100]!r}")
    return Prediction(**json.loads(m.group()))


@retry(retry=retry_if_not_exception_type(QuotaExhausted),
       wait=wait_exponential(multiplier=1, min=2, max=30),
       stop=stop_after_attempt(5), reraise=True)
def _call_with_retry(provider: LLMProvider, system: str, user: str, max_tokens: int):
    try:
        return provider.complete(system, user, max_tokens=max_tokens)
    except Exception as e:
        if _is_daily_quota(e):
            raise QuotaExhausted(str(e)[:250]) from e
        raise

def classify_one(provider: LLMProvider, text: str, system: str,
                 max_tokens: int = 250) -> dict:
    user = USER_TEMPLATE.format(text=text)

    t0 = time.perf_counter()
    raw, tin, tout = _call_with_retry(provider, system, user, max_tokens)
    latency = time.perf_counter() - t0

    row = {"text": text, "raw": raw, "input_tokens": tin,
           "output_tokens": tout, "latency_s": round(latency, 3)}
    
    try:
        p = parse(raw)
        row.update(pred=p.label, confidence=p.confidence,
                   reason=p.reason, parse_ok=True, parse_error=None)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        row.update(pred=None, confidence=None, reason=None, parse_ok=False,
                   parse_error=f"{type(e).__name__}: {str(e)[:120]}")
    return row


def classify_all(provider: LLMProvider, texts: list[str],
                 system: str = SYSTEM, max_tokens: int = 250,
                 desc: str = "", checkpoint: Path | None = None) -> pd.DataFrame:
    rows = []
    try:
        for t in tqdm(texts, desc=desc or provider.model):
            rows.append(classify_one(provider, t, system, max_tokens))
    except QuotaExhausted as e:
        print(f"\nQUOTA FILLED: {len(rows)}/{len(texts)} done.\n{e}")
    finally:
        df = pd.DataFrame(rows)
        if checkpoint is not None and len(df):
            df.to_csv(checkpoint, index=False)
            print(f"partial results: {checkpoint}")
    return df
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

import pandas as pd
from pydantic import ValidationError
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)
from tqdm import tqdm

from src.llm import LLMProvider
from src.prompts import SYSTEM, USER_TEMPLATE
from src.schema import Prediction


JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

def parse(raw: str) -> Prediction:
    m = JSON_BLOCK.search(raw or "")
    if not m:
        raise ValueError(f"JSON not found: {(raw or '')[:100]!r}")
    return Prediction(**json.loads(m.group()))


@retry(retry=retry_if_exception_type(Exception),
       wait=wait_exponential(multiplier=1, min=2, max=30),
       stop=stop_after_attempt(5), reraise=True)
def _call_with_retry(provider: LLMProvider, system: str, user: str, max_tokens: int):
    return provider.complete(system, user, max_tokens=max_tokens)


def classify_one(provider: LLMProvider, text: str, system: str,
                 max_tokens: int = 250) -> dict:
    user = USER_TEMPLATE.format(text=text)

    t0 = time.perf_counter()
    raw, tin, tout = _call_with_retry(provider, system, user, max_tokens)
    latency = time.perf_counter() - t0

    row = {"raw": raw, "input_tokens": tin, "output_tokens": tout,
           "latency_s": round(latency, 3), "text": text}
    try:
        p = parse(raw)
        row.update(pred=p.label, confidence=p.confidence,
                   reason=p.reason, parse_ok=True, parse_error=None)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        row.update(pred=None, confidence=None, reason=None,
                   parse_ok=False, parse_error=f"{type(e).__name__}: {str(e)[:120]}")
    return row


def classify_all(provider: LLMProvider, texts: list[str],
                 system: str = SYSTEM, max_tokens: int = 250,
                 desc: str = "") -> pd.DataFrame:
    rows = [classify_one(provider, t, system, max_tokens)
            for t in tqdm(texts, desc=desc or provider.model)]
    return pd.DataFrame(rows)
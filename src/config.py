from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RUNS_LOG = RESULTS / "runs.jsonl"

PREDICTIONS = RESULTS / "predictions"
ANALYSIS = RESULTS / "analysis"
FIGURES = RESULTS / "figures"

SEED = 42
LABELS = ["negative", "neutral", "positive"]

EVAL_SIZE = 1000
EVAL_BAL_SIZE = 300     # class-balanced (100/100/100)     
DEV_SIZE = 200         
POOL_SIZE = 30000    
TRAIN_SIZES = [100, 500, 2000, 10000, 30000]

# --- LLM ---
PROVIDER = "groq"   # "groq" | "anthropic"
LLM_MODEL = "openai/gpt-oss-120b"

BENCH_MODELS_FREE = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
BENCH_MODELS_PAID = ["claude-haiku-4-5-20251001"]

# for REPORTING purposes.. actual spending on the Groq free tier is $0
PRICING = {
    "openai/gpt-oss-20b":         (0.075, 0.30),
    "openai/gpt-oss-120b":        (0.15,  0.60),
    "claude-haiku-4-5-20251001":  (1.00,  3.00),
}

DAILY_TPD = {
    "openai/gpt-oss-20b":  200_000,
    "openai/gpt-oss-120b": 200_000,
}

MAX_USD_PER_RUN = 0.60
MAX_USD_TOTAL = 3.00

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, PREDICTIONS, ANALYSIS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
    
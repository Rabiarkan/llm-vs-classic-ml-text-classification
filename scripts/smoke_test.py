import sys

from dotenv import load_dotenv

load_dotenv()


def check_imports():
    import openai, anthropic, pandas, sklearn, xgboost, pydantic 
    print("✓ packages loaded")


def check_paths():
    from src.config import DATA_RAW, DATA_PROCESSED, RESULTS
    assert DATA_RAW.exists() and DATA_PROCESSED.exists() and RESULTS.exists()
    print("✓ folder structure is ready")


def check_llm():
    from src.config import PROVIDER, LLM_MODEL
    from src.llm import get_provider, assert_model_available
    from src.tracker import estimate_cost

    p = get_provider(PROVIDER, LLM_MODEL)
    assert_model_available(p)          
    print(f"✓ model accessible: {p.name}/{p.model}")

    text, tin, tout = p.complete(
        system="Reply with exactly one word, no punctuation.",
        user="Say OK",
        max_tokens=16,
    )
    cost = estimate_cost(tin, tout, p.model)
    print(f"✓ answer: {text.strip()!r}")
    print(f"  token in/out: {tin}/{tout} | list price ~${cost:.6f}")


def check_tracker():
    from src.config import LLM_MODEL
    from src.tracker import log_run, timed, estimate_cost, guard_budget

    guard_budget(n_examples=150, model=LLM_MODEL, max_usd=0.50)

    with timed() as t:
        sum(range(100_000))

    rec = log_run(
        "smoke_test",
        {"macro_f1": None},
        provider="n/a",
        model=LLM_MODEL,
        latency_p50_s=t["elapsed_s"],
        cost_usd=estimate_cost(100, 10, LLM_MODEL),
        notes="infra verification",
    )
    print(f"✓ tracker running: {rec['timestamp']}")


if __name__ == "__main__":
    try:
        check_imports()
        check_paths()
        check_llm()
        check_tracker()
    except Exception as e:
        print(f"\n✗ ERRROR: {type(e).__name__}: {e}")
        sys.exit(1)
    print("\nDone...")
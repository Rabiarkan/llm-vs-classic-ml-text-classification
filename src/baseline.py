# TF-IDF + LogisticRegression / XGBoost baseline model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED, SEED, LABELS, RESULTS
from src.tracker import log_run, timed


def build_pipeline(kind: str = "logreg") -> Pipeline:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, stop_words="english")
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=1.0,
                                 class_weight="balanced", random_state=SEED)
    else:
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                            subsample=0.9, random_state=SEED,
                            eval_metric="mlogloss")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def run(kind: str = "logreg") -> dict:
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")

    y_train, y_test = train["label"], test["label"]
    if kind == "xgb":                       
        m = {l: i for i, l in enumerate(LABELS)}
        y_train, y_test = y_train.map(m), y_test.map(m)

    pipe = build_pipeline(kind)

    with timed() as t_fit:
        pipe.fit(train["text"], y_train)
    with timed() as t_pred:
        pred = pipe.predict(test["text"])

    out = test[["text", "label"]].copy()
    out["pred"] = pred if kind == "logreg" else [LABELS[i] for i in pred]
    out.to_csv(RESULTS / f"preds_tfidf_{kind}.csv", index=False)

    print(f"\n= confusion_matrix =")
    print(pd.DataFrame(confusion_matrix(y_test, pred),
                    index=[f"true_{l}" for l in LABELS],
                    columns=[f"pred_{l}" for l in LABELS]))
    
    print(f"\n= classification_report =")
    print(classification_report(y_test, pred, digits=3))

    metrics = {
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
    }
    log_run(
        f"tfidf_{kind}", metrics,
        provider="local", model=f"tfidf+{kind}",
        n_test=len(test),
        train_time_s=t_fit["elapsed_s"],
        latency_p50_s=round(t_pred["elapsed_s"] / len(test), 6),
        cost_usd=0.0,
        notes="baseline, 350 train",
    )
    return metrics


if __name__ == "__main__":
    for k in ("logreg", "xgb"):
        print(f"\n=== {k} ===")
        run(k)
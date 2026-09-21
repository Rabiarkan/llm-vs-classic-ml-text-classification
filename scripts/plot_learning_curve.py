"""Learning Curve + LLM Reference Band
  log x-ekseni : 100→30.000 on a linear axis, the first three points are clustered;
    a log scale makes the shape of the curve visible.

  LLM as a BAND : We measured that setting `temperature=0` is not deterministic (same model,
    same prompt, a 1.6–2.5-point difference across runs). A single line hides this
    uncertainty; the band shows the true picture.

"""
import json

import matplotlib

matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ANALYSIS, FIGURES, RUNS_LOG

EVAL_SETS = ["balanced", "natural"]
COLORS = {"logreg": "#1f77b4", "xgb": "#8c8c8c",
          "llm": "#d62728", "floor": "#999999"}

def load():
    curve = pd.read_csv(ANALYSIS / "learning_curve.csv")
    with open(RUNS_LOG, encoding="utf-8") as f:
        runs = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    return curve, runs

def main() -> None:
    curve, runs = load()    
    llm_all = (runs[runs["method"].isin(["zero_shot", "few_shot"])]
           .sort_values("timestamp")
           .drop_duplicates(subset=["method", "model", "eval_set"], keep="last"))
    
    trivial = runs[runs["method"] == "trivial_majority"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, ev in zip(axes, EVAL_SETS):
        aggs = {}

        # --- baseline ---
        for kind, marker, style, label in [
            ("logreg", "o", "-", "TF-IDF + LogReg (3 seeds)"),
            ("xgb", "s", "--", "TF-IDF + XGBoost (3 seeds, ≤10k)"),
        ]:
            g = curve[(curve["eval_set"] == ev) & (curve["model"] == kind)]
            if g.empty:
                continue
            agg = g.groupby("n_train")["macro_f1"].agg(["mean", "std"]).fillna(0.0)
            aggs[kind] = agg
            ax.errorbar(agg.index, agg["mean"], yerr=agg["std"],
                        marker=marker, ls=style, capsize=4,
                        lw=2.0 if kind == "logreg" else 1.5,
                        alpha=1.0 if kind == "logreg" else 0.75,
                        color=COLORS[kind], label=label)

            if kind == "logreg":
                last_x = agg.index[-1]
                for x, y in agg["mean"].items():
                    dy = -14 if x == last_x else 9
                    ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, dy),
                                textcoords="offset points", ha="center",
                                fontsize=7, color=COLORS["logreg"])

        # --- LLM band ---
        sub = llm_all.loc[llm_all["eval_set"] == ev, "macro_f1"]
        if not sub.empty:
            lo, hi = float(sub.min()), float(sub.max())
            ax.axhspan(lo, hi, alpha=0.12, color=COLORS["llm"], zorder=0)
            ax.axhline(hi, color=COLORS["llm"], lw=2,
                       label=f"LLM, 0–4 labels (band = min–max of runs)")

        # --- trivial base ---
        t = trivial.loc[trivial["eval_set"] == ev, "macro_f1"]
        if not t.empty:
            floor = float(t.iloc[0])
            ax.axhline(floor, color=COLORS["floor"], ls=":", lw=1.5,
                       label=f"trivial floor (majority class)")

        # --- final meausure ---
        if "logreg" in aggs and len(aggs["logreg"]) >= 2:
            ys = aggs["logreg"]["mean"].to_numpy()
            ax.annotate(f"last 3× of labels: {ys[-1] - ys[-2]:+.3f}",
                        xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                        fontsize=7.5, color=COLORS["logreg"],
                        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec=COLORS["logreg"], alpha=0.9, lw=0.8))

        ax.set_xscale("log")
        ax.tick_params(labelleft=True)     
        ax.set_xlabel("labelled training examples")
        ax.set_title(f"eval_{ev}", fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_ylim(0.10, 0.88)

    axes[0].set_ylabel("macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    rank = {"LogReg": 0, "XGBoost": 1, "LLM": 2, "trivial": 3}
    order = sorted(
        range(len(labels)),
        key=lambda i: next((v for k, v in rank.items() if k in labels[i]), 9))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               framealpha=0.95)

    fig.suptitle("30k labels reach the LLM band on balanced classes — "
                 "not on the production distribution", fontsize=11.5)
    plt.tight_layout()
    

    out = FIGURES / "learning_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"record: {out}")

    # --- consol ---
    for ev in EVAL_SETS:
        g = curve[(curve["eval_set"] == ev) & (curve["model"] == "logreg")]
        if g.empty:
            continue
        agg = g.groupby("n_train")["macro_f1"].agg(["mean", "std"]).round(4)
        sub = llm_all.loc[llm_all["eval_set"] == ev, "macro_f1"]
        band = f"{sub.min():.3f}-{sub.max():.3f}" if not sub.empty else "n/a"
        ys = agg["mean"].to_numpy()
        print(f"\neval_{ev}  |  LLM band {band}  |  "
              f"final step profit {ys[-1] - ys[-2]:+.4f}")
        print(agg.to_string())


if __name__ == "__main__":
    main()
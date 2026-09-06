"""Learning Curve + LLM Reference Band
  log x-ekseni : 100→10.000 on a linear axis, the first three points are clustered;
    a log scale makes the shape of the curve visible and shows that it is NOT SATURATED.

  LLM as a BAND : We measured that setting `temperature=0` is not deterministic (same model,
    same prompt, a 1.6–2.5-point difference across runs). A single line hides this
    uncertainty; the band shows the true picture.

"""
import json

import matplotlib

matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ANALYSIS, FIGURES, RUNS_LOG

EVAL_SETS = ["balanced", "natural"]
COLORS = {"logreg": "#1f77b4", "xgb": "#8c8c8c",
          "llm": "#d62728", "floor": "#999999"}
LABEL_USD = 0.208              # cost per label (0.5 min × $25/hour)


def load():
    curve = pd.read_csv(ANALYSIS / "learning_curve.csv")
    with open(RUNS_LOG, encoding="utf-8") as f:
        runs = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    return curve, runs


def extrapolate(agg: pd.DataFrame, target: float) -> float | None:
    if len(agg) < 2:
        return None
    ys = agg["mean"].to_numpy(dtype=float)
    if ys[-1] >= target:
        return None
    xs = agg.index.to_numpy(dtype=float)
    denom = np.log10(xs[-1]) - np.log10(xs[-2])
    if denom <= 0:
        return None
    slope = (ys[-1] - ys[-2]) / denom
    if slope <= 0:
        return None
    return float(10 ** (np.log10(xs[-1]) + (target - ys[-1]) / slope))


def main() -> None:
    curve, runs = load()
    llm_all = runs[runs["method"].isin(["zero_shot", "few_shot"])]
    trivial = runs[runs["method"] == "trivial_majority"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, ev in zip(axes, EVAL_SETS):
        aggs = {}

        # --- baseline curves ---
        for kind, marker, style, label in [
            ("logreg", "o", "-", "TF-IDF + LogReg (3 seeds)"),
            ("xgb", "s", "--", "TF-IDF + XGBoost (3 seeds)"),
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
                for x, y in agg["mean"].items():
                    label_offset = 13 if ev == "balanced" else 9
                    ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, label_offset),
                                textcoords="offset points", ha="center",
                                fontsize=7, color=COLORS["logreg"])

        # --- LLM band ---
        sub = llm_all.loc[llm_all["eval_set"] == ev, "macro_f1"]
        lo = hi = None
        if not sub.empty:
            lo, hi = float(sub.min()), float(sub.max())
            ax.axhspan(lo, hi, alpha=0.12, color=COLORS["llm"], zorder=0)
            ax.axhline(hi, color=COLORS["llm"], lw=2,
                       label=f"LLM, zero labels ({lo:.2f}-{hi:.2f})")

        # --- trivial base ---
        t = trivial.loc[trivial["eval_set"] == ev, "macro_f1"]
        if not t.empty:
            floor = float(t.iloc[0])
            ax.axhline(floor, color=COLORS["floor"], ls=":", lw=1.5,
                       label=f"trivial floor ({floor:.2f})")

        # --- Extrapolation: When will it fall below the baseline band? ---
        if "logreg" in aggs and lo is not None:
            n_needed = extrapolate(aggs["logreg"], lo)
            if n_needed:
                ax.annotate(
                    f"\u2265{n_needed:,.0f} labels to reach\nthe bottom of the band",
                    xy=(0.97, 0.05), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color=COLORS["llm"],
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=COLORS["llm"], alpha=0.9, lw=0.8))

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
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower left", bbox_to_anchor=(1.0, 0.08),
               fontsize=8, framealpha=0.92)

    fig.suptitle("The curves do not cross: 10k labels still below zero-shot LLM",
                 fontsize=11.5)
    plt.tight_layout()

    out = FIGURES / "learning_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"record: {out}")

    for ev in EVAL_SETS:
        g = curve[(curve["eval_set"] == ev) & (curve["model"] == "logreg")]
        if g.empty:
            continue
        agg = g.groupby("n_train")["macro_f1"].agg(["mean", "std"]).round(4)
        sub = llm_all.loc[llm_all["eval_set"] == ev, "macro_f1"]
        band = f"{sub.min():.3f}-{sub.max():.3f}" if not sub.empty else "n/a"
        print(f"\neval_{ev}  |  LLM band {band}")
        print(agg.to_string())
        if not sub.empty:
            n = extrapolate(agg, float(sub.min()))
            if n:
                print(f" to reach under the band  >={n:,.0f} label "
                      f"(~${n * LABEL_USD:,.0f} tagging, LOWER LIMIT)")
            else:
                print("has already fallen below the baseline band or has a negative slope")


if __name__ == "__main__":
    main()
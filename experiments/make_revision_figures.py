"""Revision figures drawn at their printed size (one 3.38 in manuscript column).

Figures are saved without bbox_inches="tight" so the canvas stays exactly 3.38 in wide;
insert them in the manuscript at 100% scale and the text prints at the stated point size.

    python -m experiments.make_revision_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MM_DIR = PROJECT_ROOT / "results" / "missing_modality"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

COLUMN_WIDTH_IN = 3.38
BLUE, RED, GREEN = "#4C72B0", "#C44E52", "#55A868"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


def fig6_modality_drop():
    drop = pd.read_csv(MM_DIR / "drop_percentage_summary.csv")
    labels = [s.replace("No ", "No\n") for s in drop["Scenario"]]
    colors = [RED if s == "No ECG" else BLUE for s in drop["Scenario"]]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))
    bars = ax.bar(labels, drop["mean"], color=colors, edgecolor="black", linewidth=0.7, width=0.7)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 12.5)
    ax.set_ylabel("Relative Macro-F1 drop (%)")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.2)
    save(fig, "fig6_modality_drop")


def multi_bar(metric, xlabel, name, xlim):
    d = pd.read_csv(MM_DIR / "multi_dropout_all_subjects.csv")
    d = d.groupby("Scenario")[metric].mean().sort_values(ascending=False)
    labels = [s.replace("No ", "−").replace(" + ", ", ") for s in d.index]
    colors = [GREEN if s == "Full" else BLUE for s in d.index]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.1))
    y = range(len(d))
    ax.barh(y, d.values, color=colors, edgecolor="black", linewidth=0.6, height=0.72)
    for yi, v in zip(y, d.values):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()                                   # best scenario at the top
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.47, right=0.97, top=0.99, bottom=0.14)
    save(fig, name)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig6_modality_drop()
    multi_bar("F1", "Macro-F1", "fig8_multi_f1", (0, 1.08))
    multi_bar("RS", "Robustness Score", "fig9_multi_rs", (0, 1.12))

"""Architecture figure for the CNN and CNN-LSTM models (full manuscript width, 7.0 in).

Layer sizes are taken from models/cnn_encoder.py and models/cnn_lstm_encoder.py (the
configuration that produced the reported results).

    python -m experiments.make_architecture_figure
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 600,
})

MODALITIES = [("ECG", "#4C72B0"), ("EDA", "#DD8452"), ("Resp.", "#55A868"), ("Temp.", "#C44E52")]
MASK_COLOR = "#B22222"


def box(ax, x, y, w, h, text="", fc="#F4F4F4", ls="-", fs=7, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec="black", lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=7))


def stack(ax, x, yc, w, labels_texts, bar_h=0.2, gap=0.06):
    """Four coloured bars, one per sensor, vertically centred on yc."""
    total = 4 * bar_h + 3 * gap
    y = yc + total / 2 - bar_h
    for (name, color), text in zip(MODALITIES, labels_texts):
        ax.add_patch(Rectangle((x, y), w, bar_h, fc=color, ec="black", lw=0.6, alpha=0.85))
        ax.text(x + w / 2, y + bar_h / 2, text or name, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
        y -= bar_h + gap
    return total


def panel(ax, y0, title, enc_text, emb_dim, concat_dim, clf_text):
    yc = y0 + 0.86
    ax.text(0.05, y0 + 1.80, title, fontsize=8.5, fontweight="bold", va="center")

    # input window
    stack_h = stack(ax, 0.05, yc, 0.70, [None] * 4)
    ax.text(0.40, yc + stack_h / 2 + 0.13, "Input (4 × 3000)", ha="center", fontsize=7)
    arrow(ax, 0.77, 0.95, yc)

    # per-sensor encoders (drawn as a stack of four copies)
    enc_w, enc_h = 2.0, 1.12
    for k, off in enumerate((0.12, 0.08, 0.04)):
        box(ax, 0.95 + off, yc - enc_h / 2 + off * 0.7, enc_w, enc_h, fc="white", lw=0.6)
    box(ax, 0.95, yc - enc_h / 2, enc_w, enc_h, enc_text, fc="#EAF1FA")
    ax.text(0.95 + (enc_w + 0.12) / 2, yc + enc_h / 2 + 0.16,
            "Encoder ×4 (separate weights per sensor)", ha="center", fontsize=7)
    arrow(ax, 3.11, 3.30, yc)

    # embeddings, zeroed when a sensor is missing
    stack(ax, 3.30, yc, 0.45, [f"{emb_dim}"] * 4)
    ax.add_patch(Rectangle((3.25, yc - 0.55), 0.55, 1.10, fc="none", ec=MASK_COLOR, lw=0.9, ls="--"))
    ax.text(3.525, yc - 0.55 - 0.13, "set to 0 if\nsensor missing", ha="center", va="top",
            fontsize=6.5, color=MASK_COLOR, linespacing=1.15)
    ax.text(3.525, yc + 0.55 + 0.1, "Embeddings", ha="center", fontsize=7)
    arrow(ax, 3.80, 3.90, yc)

    # concatenation
    box(ax, 3.90, yc - 0.49, 0.50, 0.98, f"Concat\n{concat_dim}", fc="#F4F4F4")
    arrow(ax, 4.40, 4.55, yc)

    # classifier head
    box(ax, 4.55, yc - 0.49, 1.50, 0.98, clf_text, fc="#FBF0E4")
    arrow(ax, 6.05, 6.20, yc)

    # output
    box(ax, 6.20, yc - 0.25, 0.75, 0.50, "Non-stress\n/ Stress", fc="#EAF5EA")


def main():
    fig = plt.figure(figsize=(7.0, 4.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 4.15)
    ax.axis("off")

    panel(
        ax, 2.2,
        "(a) CNN  (4.41 M parameters)",
        "Conv1d 1→32, k=7, s=2\nBN + ReLU\nConv1d 32→64, k=5, s=2\nBN + ReLU\n"
        "Conv1d 64→128, k=3, s=2\nBN + ReLU\nAdaptive avg-pool → 64\nFC 8192 → 128",
        emb_dim=128, concat_dim="512",
        clf_text="FC 512→128, ReLU\nDropout 0.3\nFC 128→64, ReLU\nDropout 0.3\nFC 64→2",
    )
    ax.plot([0.05, 6.95], [2.13, 2.13], color="0.6", lw=0.6)
    panel(
        ax, 0.15,
        "(b) CNN–LSTM  (0.52 M parameters)",
        "Conv1d 1→32, k=7, s=2\nBN + ReLU\nConv1d 32→64, k=5, s=2\nBN + ReLU\n"
        "Adaptive avg-pool → 100 steps\nLSTM (hidden 64)\nFC 64 → 256",
        emb_dim=256, concat_dim="1024",
        clf_text="FC 1024→256, ReLU\nDropout 0.3\nFC 256→64, ReLU\nDropout 0.3\nFC 64→2",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_architecture.{ext}")
    print("saved fig_architecture")


if __name__ == "__main__":
    main()

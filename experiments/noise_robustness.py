"""Robustness to noisy and partially dropped sensor signals (inference only).

For each LOSO checkpoint, corrupt one modality at a time in the raw input and
measure Macro-F1 and the Robustness Score (RS = F1_corrupted / F1_clean).

Corruptions per modality:
  - Gaussian noise at SNR 20, 10 and 0 dB
  - Contiguous segment dropout removing 25% and 50% of the window (set to zero)

Run from the project root:
    python -m experiments.noise_robustness
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# The saved LOSO checkpoints were trained with the default (larger) configuration in
# cnn_lstm_encoder.py; they reproduce the reported full-modality Macro-F1 of 0.879.
from models.cnn_lstm_encoder import MultiModalCNNLSTM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_ds"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "missing_modality"

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
MODALITY_NAMES = ["ECG", "EDA", "Respiration", "Temperature"]
SNR_DB = [20, 10, 0]
DROP_FRACTIONS = [0.25, 0.50]
SEED = 42
N_DRAWS = 5      # independent random corruptions per condition, averaged per subject


def load_subject(sid):
    X = np.load(DATA_DIR / f"S{sid}_X.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"S{sid}_y.npy").astype(np.int64)
    return X, y


def load_model(sid):
    model = MultiModalCNNLSTM()
    model.load_state_dict(torch.load(MODEL_DIR / f"cnnlstm_S{sid}.pt", map_location="cpu"))
    model.eval()
    return model


def add_noise(X, mod, snr_db, rng):
    """Add Gaussian noise to one modality, scaled to the window's own signal power."""
    X = X.copy()
    sig = X[:, mod, :]
    sig_std = sig.std(axis=1, keepdims=True)
    noise_std = sig_std / (10 ** (snr_db / 20))
    X[:, mod, :] = sig + rng.standard_normal(sig.shape).astype(np.float32) * noise_std
    return X


def drop_segment(X, mod, fraction, rng):
    """Zero one contiguous segment of a modality in every window, at a random position."""
    X = X.copy()
    n_windows, _, length = X.shape
    seg_len = int(round(fraction * length))
    starts = rng.integers(0, length - seg_len + 1, size=n_windows)
    for i, start in enumerate(starts):
        X[i, mod, start:start + seg_len] = 0.0
    return X


def evaluate(model, X, y):
    probs, preds = [], []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            logits = model(torch.tensor(X[start:start + 64]))
            probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
            preds.append(logits.argmax(dim=1).numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    return {
        "Accuracy": accuracy_score(y, preds),
        "F1": f1_score(y, preds, average="macro", zero_division=0),
        "AUROC": roc_auc_score(y, probs),
    }


def conditions():
    yield "Clean", None, None
    for mod, name in enumerate(MODALITY_NAMES):
        for snr in SNR_DB:
            yield f"{name} noise {snr} dB", mod, ("noise", snr)
        for frac in DROP_FRACTIONS:
            yield f"{name} dropout {int(frac * 100)}%", mod, ("drop", frac)


def corrupt(X, mod, spec, rng):
    if spec is None:
        return X
    kind, value = spec
    if kind == "noise":
        return add_noise(X, mod, value, rng)
    return drop_segment(X, mod, value, rng)


def main():
    rows = []
    for sid in SUBJECTS:
        X, y = load_subject(sid)
        model = load_model(sid)
        rng = np.random.default_rng(SEED + sid)

        clean_f1 = None
        for label, mod, spec in conditions():
            for draw in range(1 if spec is None else N_DRAWS):
                Xc = corrupt(X, mod, spec, rng)
                result = evaluate(model, Xc, y)
                if label == "Clean":
                    clean_f1 = result["F1"]
                rows.append({
                    "Subject": sid,
                    "Condition": label,
                    "Draw": draw,
                    **result,
                    "RS": result["F1"] / clean_f1,
                })
        print(f"S{sid} done")

    per_subject = pd.DataFrame(rows)
    per_subject_mean = (                      # average the draws within each subject first
        per_subject
        .drop(columns="Draw")
        .groupby(["Condition", "Subject"], sort=False)
        .mean()
        .reset_index()
    )
    summary = (
        per_subject_mean
        .groupby("Condition", sort=False)
        .agg(
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            RS_mean=("RS", "mean"),
            RS_std=("RS", "std"),
            AUROC_mean=("AUROC", "mean"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(OUTPUT_DIR / "noise_robustness_per_subject.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "noise_robustness_summary.csv", index=False)

    print("\nClean Macro-F1 (should match the paper's 0.879):",
          round(summary.loc[summary.Condition == "Clean", "F1_mean"].item(), 3))
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

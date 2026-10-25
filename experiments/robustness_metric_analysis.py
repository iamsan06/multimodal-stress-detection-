"""How the Robustness Score (RS) relates to other robustness measures (reviewer comment 6).

For each dropout scenario of the CNN-LSTM, compare:
  - mean RS (mean over subjects of F1_missing / F1_full), as reported in the paper
  - relative drop (%) = (1 - RS) x 100, which is the same information
  - mean absolute Macro-F1 drop
  - ratio of means, mean(F1_missing) / mean(F1_full)
  - median RS and worst-case (minimum) RS over subjects
  - chance-normalised RS: (F1_missing - F1_floor) / (F1_full - F1_floor), where F1_floor is the
    Macro-F1 of always predicting the majority class on that subject

    python -m experiments.robustness_metric_analysis
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MM_DIR = PROJECT_ROOT / "results" / "missing_modality"
DATA_DIR = PROJECT_ROOT / "data" / "processed_ds"


def majority_floor(sid):
    y = np.load(DATA_DIR / f"S{sid}_y.npy")
    majority = np.bincount(y).argmax()
    return f1_score(y, np.full_like(y, majority), average="macro", zero_division=0)


def analyse(df):
    floor = {sid: majority_floor(sid) for sid in df.Subject.unique()}
    full = df[df.Scenario == "Full"].set_index("Subject").F1
    rows = []
    for scenario in df.Scenario.unique():
        if scenario == "Full":
            continue
        d = df[df.Scenario == scenario].set_index("Subject")
        f1_full = full.loc[d.index]
        fl = pd.Series(floor).loc[d.index]
        rs = d.F1 / f1_full
        norm = (d.F1 - fl) / (f1_full - fl)
        rows.append({
            "Scenario": scenario,
            "RS_mean": rs.mean(),
            "RelDrop_pct": (1 - rs.mean()) * 100,
            "AbsDrop_F1": (f1_full - d.F1).mean(),
            "RatioOfMeans": d.F1.mean() / f1_full.mean(),
            "RS_median": rs.median(),
            "RS_worst": rs.min(),
            "Subjects_RS_gt_1": int((rs > 1).sum()),
            "ChanceNorm_RS_mean": norm.mean(),
            "ChanceNorm_RS_median": norm.median(),
        })
    return pd.DataFrame(rows)


def main():
    single = analyse(pd.read_csv(MM_DIR / "single_dropout_all_subjects.csv"))
    multi = analyse(pd.read_csv(MM_DIR / "multi_dropout_all_subjects.csv"))
    single.to_csv(MM_DIR / "robustness_metric_comparison_single.csv", index=False)
    multi.to_csv(MM_DIR / "robustness_metric_comparison_multi.csv", index=False)
    floors = pd.Series({sid: majority_floor(sid) for sid in range(2, 18) if sid != 12})
    print(f"Majority-class Macro-F1 floor: mean {floors.mean():.3f} (range {floors.min():.3f}-{floors.max():.3f})\n")
    pd.set_option("display.width", 220)
    print(single.round(3).to_string(index=False))
    print()
    print(multi.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

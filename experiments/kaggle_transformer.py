# Kaggle notebook: per-modality Transformer baseline, LOSO (item 5).
# Each "# %%" block is one notebook cell. Attach the wesad_moddrop_dataset Kaggle dataset
# (processed_ds + baseline), turn on a GPU accelerator, and run the cells in order.

# %% Cell 1 - imports and device
import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% Cell 2 - paths and settings
INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = next(INPUT_ROOT.rglob("processed_ds"))
BASE_DIR = next(INPUT_ROOT.rglob("baseline"))

EXPORT = Path("/kaggle/working/export_tf")        # everything inside is zipped at the end
CSV_DIR = EXPORT / "results" / "transformer"
CKPT_DIR = EXPORT / "results" / "models_transformer"
FOLD_DIR = Path("/kaggle/working/fold_cache_tf")  # per-fold results, lets a crashed run resume
for d in (CSV_DIR, CKPT_DIR, FOLD_DIR):
    d.mkdir(parents=True, exist_ok=True)

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
VAL_SUBJECT, VAL_FALLBACK = 17, 16               # same validation subjects as the other models
EPOCHS, PATIENCE, LR, WEIGHT_DECAY = 30, 8, 1e-3, 1e-4
SEED = 42

# Transformer size: chosen so the whole model is close to the CNN-LSTM (~522k parameters)
D_MODEL, N_HEAD, D_FF, N_LAYERS = 48, 4, 96, 2
PATCH = 30                                       # 3000 samples / 30 = 100 tokens per modality
EMBED_DIM = 256                                  # per-modality embedding, same as the CNN-LSTM
print("Data:", DATA_DIR, "| Baseline:", BASE_DIR)

# %% Cell 3 - load all subjects into memory
def load_subject(sid):
    X = np.load(DATA_DIR / f"S{sid}_X.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"S{sid}_y.npy").astype(np.int64)
    return X, y


DATA = {sid: load_subject(sid) for sid in SUBJECTS}
print({sid: DATA[sid][0].shape for sid in SUBJECTS[:3]}, "...")

# %% Cell 4 - model: one Transformer encoder per modality, same fusion head as the CNN-LSTM
class ModalityTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=N_HEAD, d_ff=D_FF, layers=N_LAYERS,
                 patch=PATCH, seq_len=3000, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, seq_len // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                                   # x: (batch, 1, 3000)
        tokens = self.patch_embed(x).transpose(1, 2) + self.pos
        return self.norm(self.encoder(tokens)).mean(dim=1)  # (batch, d_model)


class MultiModalTransformer(nn.Module):
    def __init__(self, n_modalities=4, embed_dim=EMBED_DIM, n_classes=2):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([ModalityTransformer() for _ in range(n_modalities)])
        self.embed_proj = nn.ModuleList([nn.Linear(D_MODEL, embed_dim) for _ in range(n_modalities)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, missing_mask=None):
        embeddings = []
        for i in range(self.n_modalities):
            emb = self.embed_proj[i](self.encoders[i](x[:, i:i + 1, :]))
            if missing_mask is not None and missing_mask[i]:
                emb = torch.zeros_like(emb)                 # same zero-masking as the CNN-LSTM
            embeddings.append(emb)
        return self.classifier(torch.cat(embeddings, dim=1))


print("Parameters:", sum(p.numel() for p in MultiModalTransformer().parameters()),
      "(CNN-LSTM: 521,666)")

# %% Cell 5 - prediction and training (same settings as the baseline LOSO runs)
@torch.no_grad()
def predict(model, X, missing_mask=None, batch_size=64):
    model.eval()
    preds, probs = [], []
    for s in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[s:s + batch_size]).to(device), missing_mask=missing_mask)
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def train_fold(model, X_tr, y_tr, X_val, y_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    counts = np.bincount(y_tr)
    weights = (1.0 / counts) / (1.0 / counts).sum() * len(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=32, shuffle=True)

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        val_pred, _ = predict(model, X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1, epoch + 1

# %% Cell 6 - dropout scenarios and evaluation (identical to the CNN-LSTM notebook)
SINGLE = {
    "No ECG": [True, False, False, False],
    "No EDA": [False, True, False, False],
    "No Respiration": [False, False, True, False],
    "No Temperature": [False, False, False, True],
}
MULTI = {
    "No ECG + No EDA": [True, True, False, False],
    "No ECG + No Respiration": [True, False, True, False],
    "No ECG + No Temperature": [True, False, False, True],
    "No EDA + No Respiration": [False, True, True, False],
    "No EDA + No Temperature": [False, True, False, True],
    "No Respiration + No Temperature": [False, False, True, True],
    "Only ECG": [False, True, True, True],
    "Only EDA": [True, False, True, True],
    "Only Respiration": [True, True, False, True],
    "Only Temperature": [True, True, True, False],
}


def score(y, mask, model, X):
    preds, probs = predict(model, X, missing_mask=mask)
    return {
        "Accuracy": accuracy_score(y, preds),
        "F1": f1_score(y, preds, average="macro", zero_division=0),
        "AUROC": roc_auc_score(y, probs),
    }


def evaluate_subject(model, sid):
    X, y = DATA[sid]
    full = score(y, None, model, X)
    single = [{"Subject": sid, "Scenario": "Full", **full, "RS": 1.0}]
    multi = [{"Subject": sid, "Scenario": "Full", **full, "RS": 1.0}]
    for rows, scenarios in ((single, SINGLE), (multi, MULTI)):
        for name, mask in scenarios.items():
            r = score(y, mask, model, X)
            rows.append({"Subject": sid, "Scenario": name, **r, "RS": r["F1"] / full["F1"]})
    return pd.DataFrame(single), pd.DataFrame(multi)

# %% Cell 7 - leave-one-subject-out training (safe to re-run: finished folds are skipped)
for sid in SUBJECTS:
    ckpt = CKPT_DIR / f"transformer_S{sid}.pt"
    if ckpt.exists() and (FOLD_DIR / f"S{sid}_multi.csv").exists():
        print(f"S{sid}: already done, skipping")
        continue

    val_sid = VAL_FALLBACK if sid == VAL_SUBJECT else VAL_SUBJECT
    train_ids = [s for s in SUBJECTS if s not in (sid, val_sid)]
    X_tr = np.concatenate([DATA[s][0] for s in train_ids])
    y_tr = np.concatenate([DATA[s][1] for s in train_ids])
    X_val, y_val = DATA[val_sid]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = MultiModalTransformer().to(device)

    t0 = time.time()
    model, best_val, n_epochs = train_fold(model, X_tr, y_tr, X_val, y_val)
    elapsed = time.time() - t0
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)

    single, multi = evaluate_subject(model, sid)
    single.to_csv(FOLD_DIR / f"S{sid}_single.csv", index=False)
    pd.DataFrame([{"Subject": sid, "val_subject": val_sid, "val_f1": best_val,
                   "epochs": n_epochs, "train_time_sec": elapsed}]).to_csv(FOLD_DIR / f"S{sid}_info.csv", index=False)
    multi.to_csv(FOLD_DIR / f"S{sid}_multi.csv", index=False)       # written last: marks the fold as done
    full_f1 = single.loc[single.Scenario == "Full", "F1"].item()
    print(f"S{sid}: val F1 {best_val:.3f} | epochs {n_epochs} | test F1 {full_f1:.3f} | {elapsed:.0f}s")

# %% Cell 8 - combine folds, Table II row, and comparison with the CNN-LSTM
single_all = pd.concat([pd.read_csv(FOLD_DIR / f"S{s}_single.csv") for s in SUBJECTS], ignore_index=True)
multi_all = pd.concat([pd.read_csv(FOLD_DIR / f"S{s}_multi.csv") for s in SUBJECTS], ignore_index=True)
info_all = pd.concat([pd.read_csv(FOLD_DIR / f"S{s}_info.csv") for s in SUBJECTS], ignore_index=True)
single_all.to_csv(CSV_DIR / "transformer_single_dropout_all_subjects.csv", index=False)
multi_all.to_csv(CSV_DIR / "transformer_multi_dropout_all_subjects.csv", index=False)

full = single_all[single_all.Scenario == "Full"][["Subject", "Accuracy", "F1", "AUROC"]]
full = full.merge(info_all, on="Subject")
full.to_csv(CSV_DIR / "transformer_full_modality_results.csv", index=False)

base_single = pd.read_csv(next(BASE_DIR.rglob("single_dropout_all_subjects.csv")))
base_full = base_single[base_single.Scenario == "Full"]
print("Table II rows (mean +/- std over 15 LOSO folds)")
for name, d in (("CNN-LSTM", base_full), ("Transformer", full)):
    print(f"{name:12s} Acc {d.Accuracy.mean():.3f} +/- {d.Accuracy.std():.3f} | "
          f"Macro-F1 {d.F1.mean():.3f} +/- {d.F1.std():.3f} | AUROC {d.AUROC.mean():.3f}")
print(f"Mean epochs {info_all.epochs.mean():.1f}, mean train time {info_all.train_time_sec.mean():.0f}s per fold\n")


def compare(new, base_file):
    base = pd.read_csv(next(BASE_DIR.rglob(base_file)))
    base = base[base.Subject.isin(new.Subject.unique())]
    rows = []
    for scenario in new.Scenario.unique():
        n = new[new.Scenario == scenario].sort_values("Subject")
        b = base[base.Scenario == scenario].sort_values("Subject")
        rows.append({
            "Scenario": scenario,
            "F1_cnnlstm": b.F1.mean(), "F1_transformer": n.F1.mean(),
            "RS_cnnlstm": b.RS.mean(), "RS_transformer": n.RS.mean(),
            "F1_diff": n.F1.mean() - b.F1.mean(),
            "p_value": ttest_rel(n.F1.values, b.F1.values).pvalue,
        })
    return pd.DataFrame(rows)


cmp_single = compare(single_all, "single_dropout_all_subjects.csv")
cmp_multi = compare(multi_all, "multi_dropout_all_subjects.csv")
cmp_single.to_csv(CSV_DIR / "transformer_vs_cnnlstm_single.csv", index=False)
cmp_multi.to_csv(CSV_DIR / "transformer_vs_cnnlstm_multi.csv", index=False)
print(cmp_single.round(3).to_string(index=False))
print()
print(cmp_multi.round(3).to_string(index=False))

# %% Cell 9 - zip everything for download
archive = shutil.make_archive("/kaggle/working/item5_results", "zip", root_dir=EXPORT)
print(archive, round(Path(archive).stat().st_size / 1e6, 1), "MB")

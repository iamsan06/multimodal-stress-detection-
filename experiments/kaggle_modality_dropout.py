# Kaggle notebook: CNN-LSTM trained WITH modality dropout (item 8).
# Each "# %%" block is one notebook cell. Upload wesad_moddrop_dataset.zip as a Kaggle
# dataset, turn on a GPU accelerator, and run the cells in order.

# %% Cell 1 - imports and device
import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% Cell 2 - paths and settings
INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = next(INPUT_ROOT.rglob("processed_ds"))
BASE_DIR = next(INPUT_ROOT.rglob("baseline"))

EXPORT = Path("/kaggle/working/export")          # everything inside is zipped at the end
CSV_DIR = EXPORT / "results" / "modality_dropout"
CKPT_DIR = EXPORT / "results" / "models_moddrop"
FOLD_DIR = Path("/kaggle/working/fold_cache")     # per-fold results, lets a crashed run resume
for d in (CSV_DIR, CKPT_DIR, FOLD_DIR):
    d.mkdir(parents=True, exist_ok=True)

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
VAL_SUBJECT, VAL_FALLBACK = 17, 16               # same validation subjects as the baseline
MOD_DROP_P = 0.2                                 # chance each modality is switched off per training sample
EPOCHS, PATIENCE, LR, WEIGHT_DECAY = 30, 8, 1e-3, 1e-4
SEED = 42
print("Data:", DATA_DIR, "| Baseline:", BASE_DIR)

# %% Cell 3 - load all subjects into memory
def load_subject(sid):
    X = np.load(DATA_DIR / f"S{sid}_X.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"S{sid}_y.npy").astype(np.int64)
    return X, y


DATA = {sid: load_subject(sid) for sid in SUBJECTS}
print({sid: DATA[sid][0].shape for sid in SUBJECTS[:3]}, "...")

# %% Cell 4 - model (same architecture and layer names as models/cnn_lstm_encoder.py)
class ModalityEncoderSeq(nn.Module):
    def __init__(self, conv_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, conv_channels, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x.permute(0, 2, 1)


class MultiModalCNNLSTM(nn.Module):
    def __init__(self, n_modalities=4, conv_channels=64, lstm_hidden=64, embed_dim=256,
                 n_classes=2, pooled_seq_len=100, mod_drop_p=0.0):
        super().__init__()
        self.n_modalities = n_modalities
        self.mod_drop_p = mod_drop_p
        self.encoders = nn.ModuleList([ModalityEncoderSeq(conv_channels) for _ in range(n_modalities)])
        self.seq_pool = nn.AdaptiveAvgPool1d(pooled_seq_len)
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
            for _ in range(n_modalities)
        ])
        self.embed_proj = nn.ModuleList([nn.Linear(lstm_hidden, embed_dim) for _ in range(n_modalities)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, missing_mask=None):
        embeddings = []
        for i in range(self.n_modalities):
            seq = self.encoders[i](x[:, i:i + 1, :])
            seq = self.seq_pool(seq.permute(0, 2, 1)).permute(0, 2, 1)
            _, (h_n, _) = self.lstms[i](seq)
            emb = self.embed_proj[i](h_n.squeeze(0))
            if missing_mask is not None and missing_mask[i]:
                emb = torch.zeros_like(emb)
            embeddings.append(emb)
        embs = torch.stack(embeddings, dim=1)                       # (batch, modalities, embed_dim)

        if self.training and self.mod_drop_p > 0:                   # modality dropout, training only
            drop = torch.rand(embs.size(0), self.n_modalities, device=embs.device) < self.mod_drop_p
            all_dropped = drop.all(dim=1)                           # never drop every modality at once
            if all_dropped.any():
                keep = torch.randint(0, self.n_modalities, (int(all_dropped.sum()),), device=embs.device)
                drop[all_dropped, keep] = False
            embs = embs * (~drop).unsqueeze(-1).float()

        return self.classifier(embs.flatten(1))


print("Parameters:", sum(p.numel() for p in MultiModalCNNLSTM().parameters()))

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

        val_pred, _ = predict(model, X_val)                          # validation uses all four sensors
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1, epoch + 1

# %% Cell 6 - dropout scenarios and evaluation (identical to the baseline notebook)
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
    ckpt = CKPT_DIR / f"cnnlstm_moddrop_S{sid}.pt"
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
    model = MultiModalCNNLSTM(mod_drop_p=MOD_DROP_P).to(device)

    t0 = time.time()
    model, best_val, n_epochs = train_fold(model, X_tr, y_tr, X_val, y_val)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)

    single, multi = evaluate_subject(model, sid)
    single.to_csv(FOLD_DIR / f"S{sid}_single.csv", index=False)
    multi.to_csv(FOLD_DIR / f"S{sid}_multi.csv", index=False)
    full_f1 = single.loc[single.Scenario == "Full", "F1"].item()
    print(f"S{sid}: val F1 {best_val:.3f} | epochs {n_epochs} | test F1 {full_f1:.3f} | {time.time() - t0:.0f}s")

# %% Cell 8 - combine folds and compare with the baseline model
single_all = pd.concat([pd.read_csv(FOLD_DIR / f"S{s}_single.csv") for s in SUBJECTS], ignore_index=True)
multi_all = pd.concat([pd.read_csv(FOLD_DIR / f"S{s}_multi.csv") for s in SUBJECTS], ignore_index=True)
single_all.to_csv(CSV_DIR / "moddrop_single_dropout_all_subjects.csv", index=False)
multi_all.to_csv(CSV_DIR / "moddrop_multi_dropout_all_subjects.csv", index=False)


def compare(new, base_file):
    base = pd.read_csv(next(BASE_DIR.rglob(base_file)))
    base = base[base.Subject.isin(new.Subject.unique())]
    rows = []
    for scenario in new.Scenario.unique():
        n = new[new.Scenario == scenario].sort_values("Subject")
        b = base[base.Scenario == scenario].sort_values("Subject")
        rows.append({
            "Scenario": scenario,
            "F1_baseline": b.F1.mean(), "F1_moddrop": n.F1.mean(),
            "RS_baseline": b.RS.mean(), "RS_moddrop": n.RS.mean(),
            "F1_gain": n.F1.mean() - b.F1.mean(),
            "p_value": ttest_rel(n.F1.values, b.F1.values).pvalue,
        })
    return pd.DataFrame(rows)


cmp_single = compare(single_all, "single_dropout_all_subjects.csv")
cmp_multi = compare(multi_all, "multi_dropout_all_subjects.csv")
cmp_single.to_csv(CSV_DIR / "moddrop_vs_baseline_single.csv", index=False)
cmp_multi.to_csv(CSV_DIR / "moddrop_vs_baseline_multi.csv", index=False)
print(cmp_single.round(3).to_string(index=False))
print()
print(cmp_multi.round(3).to_string(index=False))

# %% Cell 9 - zip everything for download
archive = shutil.make_archive("/kaggle/working/item8_results", "zip", root_dir=EXPORT)
print(archive, round(Path(archive).stat().st_size / 1e6, 1), "MB")

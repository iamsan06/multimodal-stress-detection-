import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
import copy

class WESADDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

DATA_DIR = "../data/processed_ds"
SUBJECTS = [s for s in range(2, 18) if s != 12]

FIXED_VAL_SUBJECT = 17
SECONDARY_VAL_SUBJECT = 16

def load_subject(sid):
    X = np.load(f"{DATA_DIR}/S{sid}_X.npy")
    y = np.load(f"{DATA_DIR}/S{sid}_y.npy")
    return X, y

def get_val_subject(test_sid):
    if test_sid == FIXED_VAL_SUBJECT:
        return SECONDARY_VAL_SUBJECT
    return FIXED_VAL_SUBJECT

def load_all_except(test_sid, val_sid):
    Xs, ys = [], []
    for sid in SUBJECTS:
        if sid == test_sid or sid == val_sid:
            continue
        X, y = load_subject(sid)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)

def train_one_fold(model, train_loader, val_loader, device, epochs=50, patience=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    all_y = torch.cat([y for _, y in train_loader])
    class_counts = torch.bincount(all_y)
    class_weights = (1.0 / class_counts.float())
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds); val_true.extend(yb.numpy())
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_f1
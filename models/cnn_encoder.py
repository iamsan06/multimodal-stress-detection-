import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    """Encodes a single-channel 1D signal (3000 samples) into a 128-dim embedding."""
    def __init__(self, embed_dim=128):  # Reduced from 256
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(64)
        self.fc    = nn.Linear(128 * 64, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)              # (batch, 128, 64)
        x = x.flatten(1)              # (batch, 128*64)
        x = self.fc(x)                # (batch, 128)
        return x

class MultiModalCNN(nn.Module):
    def __init__(self, n_modalities=4, embed_dim=128, n_classes=2):  # Reduced default
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([ModalityEncoder(embed_dim) for _ in range(n_modalities)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 128), nn.ReLU(), nn.Dropout(0.3),  # Reduced from 256
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, missing_mask=None):
        embeddings = []
        for i, encoder in enumerate(self.encoders):
            emb = encoder(x[:, i:i+1, :])
            if missing_mask is not None and missing_mask[i]:
                emb = torch.zeros_like(emb)
            embeddings.append(emb)
        fused = torch.cat(embeddings, dim=1)
        return self.classifier(fused)
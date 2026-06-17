import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoderSeq(nn.Module):
    def __init__(self, conv_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, conv_channels, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(conv_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        return x

class MultiModalCNNLSTM(nn.Module):
    def __init__(self, n_modalities=4, conv_channels=32, lstm_hidden=32, embed_dim=64, n_classes=2, pooled_seq_len=50):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([ModalityEncoderSeq(conv_channels) for _ in range(n_modalities)])
        self.seq_pool = nn.AdaptiveAvgPool1d(pooled_seq_len)
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
            for _ in range(n_modalities)
        ])
        self.embed_proj = nn.ModuleList([
            nn.Linear(lstm_hidden, embed_dim) for _ in range(n_modalities)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, missing_mask=None):
        embeddings = []
        for i in range(self.n_modalities):
            seq = self.encoders[i](x[:, i:i+1, :])
            seq = seq.permute(0, 2, 1)
            seq = self.seq_pool(seq)
            seq = seq.permute(0, 2, 1)
            _, (h_n, _) = self.lstms[i](seq)
            h = h_n.squeeze(0)
            emb = self.embed_proj[i](h)
            if missing_mask is not None and missing_mask[i]:
                emb = torch.zeros_like(emb)
            embeddings.append(emb)
        fused = torch.cat(embeddings, dim=1)
        return self.classifier(fused)
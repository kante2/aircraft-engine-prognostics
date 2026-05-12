"""Attention-LSTM for RUL Prediction"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """어느 타임스텝이 RUL 예측에 중요한지 가중치 학습"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)   # (batch, seq_len)
        weights = F.softmax(scores, dim=1)          # (batch, seq_len)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return context, weights


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, self.attn_weights = self.attention(lstm_out)
        return self.fc(context).squeeze(-1)

"""CNN-LSTM for RUL Prediction"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        # CNN: 센서 간 공간적 패턴 추출
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        # LSTM: 시간적 패턴 추출
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Conv1d expects (batch, channels, seq_len)
        c = self.cnn(x.permute(0, 2, 1))     # (batch, 64, seq_len)
        c = c.permute(0, 2, 1)                # (batch, seq_len, 64)
        lstm_out, _ = self.lstm(c)             # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)

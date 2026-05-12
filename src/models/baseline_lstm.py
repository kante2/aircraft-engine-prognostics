"""Baseline LSTM for RUL Prediction"""
import torch
import torch.nn as nn


# 네트워크 설정 
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,   # 18 (센서 개수)
            hidden_size=hidden_dim, # 64 (기억 용량) - 크면 많이 기억하지만 과적합 위험, 작으면 기억이 부족하지만 일반화 좋음
            num_layers=num_layers,  # LSTM 2층 쌓기
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
        lstm_out, _ = self.lstm(x)              # (batch, 30, 64) — 매 시점마다 기억 64개
        last_hidden = lstm_out[:, -1, :]        # (batch, 64) — 마지막 시점만 꺼냄
        out = self.fc(last_hidden).squeeze(-1)  # (batch,) — RUL 예측값
        return out

'''
30사이클 센서 데이터를 시간 순서대로 하나씩 읽어간다. (window_size=30)
 - 최근 30번 비행의 센서 기록을 묶어서 한 번에 보여준다

예)
윈도우 크기 30 = 최근 30번 비행 기록을 한 묶음으로

[비행1] 온도=620, 압력=38, RPM=2388 ...
[비행2] 온도=620, 압력=38, RPM=2388 ...
[비행3] 온도=621, 압력=38, RPM=2387 ...
...
[비행30] 온도=635, 압력=36, RPM=2380 ...  ← "온도가 올라가고 있네"

→ 이 묶음 하나가 모델의 입력 1개
→ 모델: "이 추이를 보니 RUL = 47 사이클 남았다"


매 시점마다 "지금까지 본 것을 기억"하면서 다음 시점을 읽는다. 
이전 시점의 기억을 다음 시점으로 넘긴다. (hidden state)
=>
센서 18개를 그냥 숫자 나열로 받아들임. "sensor_3이 올라가면서 
sensor_7이 내려가는 조합"같은 센서 간 관계를 직접 잡아내기 어렵다.

'''
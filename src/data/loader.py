"""PyTorch Dataset & DataLoader"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(processed_dir="data/processed", subset="FD001", batch_size=64, val_ratio=0.2):
    processed_dir = Path(processed_dir)

    X_train = np.load(processed_dir / f"X_train_{subset}.npy")
    y_train = np.load(processed_dir / f"y_train_{subset}.npy")
    X_test = np.load(processed_dir / f"X_test_{subset}.npy")
    y_test = np.load(processed_dir / f"y_test_{subset}.npy")

    # Train/Val split (엔진 단위 아닌 샘플 단위 — FD001 표준 관행)
    n = len(X_train)
    idx = np.random.permutation(n)
    val_size = int(n * val_ratio)

    X_val, y_val = X_train[idx[:val_size]], y_train[idx[:val_size]]
    X_tr, y_tr = X_train[idx[val_size:]], y_train[idx[val_size:]]

    train_loader = DataLoader(CMAPSSDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CMAPSSDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CMAPSSDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return train_loader, val_loader, test_loader

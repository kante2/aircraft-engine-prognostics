"""Training Loop"""
import torch
import numpy as np
from pathlib import Path
import time


class Trainer:
    def __init__(self, model, criterion, optimizer, device, checkpoint_dir="results/checkpoints"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_rmse": []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(X)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        total_loss = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            total_loss += self.criterion(pred, y).item() * len(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        loss = total_loss / len(loader.dataset)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        return loss, rmse, preds, targets

    def fit(self, train_loader, val_loader, epochs=50, patience=10, model_name="model"):
        best_rmse = float("inf")
        patience_counter = 0

        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val RMSE':>9} | {'Time':>6}")
        print("-" * 55)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmse, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(val_rmse)

            improved = ""
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / f"{model_name}_best.pt")
                improved = " ★"
            else:
                patience_counter += 1

            print(f"{epoch:>6} | {train_loss:>11.4f} | {val_loss:>9.4f} | {val_rmse:>9.2f} | {elapsed:>5.1f}s{improved}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

        print(f"\nBest Val RMSE: {best_rmse:.2f}")
        return self.history

"""Train & Compare All Models on any subset"""
import sys
sys.path.insert(0, "/workspace")

import torch
import numpy as np
import json
import argparse
from src.data.loader import get_dataloaders
from src.models.baseline_lstm import BaselineLSTM
from src.models.cnn_lstm import CNNLSTM
from src.models.attention_lstm import AttentionLSTM
from src.losses.asymmetric import AsymmetricMSELoss
from src.training.trainer import Trainer
from src.utils.seed import set_seed
import pickle
from pathlib import Path

MODELS = {
    "baseline_lstm": BaselineLSTM,
    "cnn_lstm": CNNLSTM,
    "attention_lstm": AttentionLSTM,
}


def train_model(name, model_cls, input_dim, train_loader, val_loader, test_loader, device, subset):
    print(f"\n{'='*60}")
    print(f"  {name} ({subset})")
    print(f"{'='*60}")

    set_seed(42)
    model = model_cls(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    criterion = AsymmetricMSELoss(alpha_early=1.0, alpha_late=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    trainer = Trainer(model, criterion, optimizer, device)
    model_tag = f"{name}_{subset}"
    history = trainer.fit(train_loader, val_loader, epochs=50, patience=10, model_name=model_tag)

    # Load best & test
    model.load_state_dict(torch.load(f"results/checkpoints/{model_tag}_best.pt"))
    _, test_rmse, preds, targets = trainer.evaluate(test_loader)

    # NASA Score
    diff = preds - targets
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    nasa_score = float(score.sum())

    print(f"\n📊 {name} — Test RMSE: {test_rmse:.2f}, NASA Score: {nasa_score:.0f}")
    return {
        "rmse": round(float(test_rmse), 2),
        "nasa_score": round(nasa_score, 0),
        "params": params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", help="FD001, FD002, FD003, FD004")
    args = parser.parse_args()
    subset = args.subset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 피처 수 자동 감지
    meta_path = Path(f"data/processed/meta_{subset}.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    input_dim = meta["n_features"]
    print(f"Subset: {subset}, Features: {input_dim}")

    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(subset=subset)

    results = {}
    for name, model_cls in MODELS.items():
        results[name] = train_model(
            name, model_cls, input_dim,
            train_loader, val_loader, test_loader,
            device, subset
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"  {subset} COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'RMSE':>8} {'NASA Score':>12} {'Params':>10}")
    print("-" * 52)
    for name, r in results.items():
        print(f"{name:<20} {r['rmse']:>8.2f} {r['nasa_score']:>12.0f} {r['params']:>10,}")

    # Save
    out_path = f"results/metrics/comparison_{subset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    main()

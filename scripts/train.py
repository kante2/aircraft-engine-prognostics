"""Training Entry Point"""
import sys
sys.path.insert(0, "/workspace")

import torch
import numpy as np
import yaml
from src.data.loader import get_dataloaders
from src.models.baseline_lstm import BaselineLSTM
from src.losses.asymmetric import AsymmetricMSELoss
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def main(config_path="configs/baseline.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        subset=cfg["data"]["subset"],
        batch_size=cfg["data"]["batch_size"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    # Model
    model = BaselineLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & Optimizer
    criterion = AsymmetricMSELoss(alpha_early=1.0, alpha_late=3.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Train
    trainer = Trainer(model, criterion, optimizer, device)
    history = trainer.fit(
        train_loader, val_loader,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        model_name=cfg["model"]["name"],
    )

    # Test
    model.load_state_dict(torch.load("results/checkpoints/baseline_lstm_best.pt"))
    test_loss, test_rmse, preds, targets = trainer.evaluate(test_loader)
    print(f"\n📊 Test RMSE: {test_rmse:.2f}")

    # NASA Score
    diff = preds - targets
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    print(f"📊 NASA Score: {score.sum():.0f}")


if __name__ == "__main__":
    main()

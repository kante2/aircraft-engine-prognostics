"""
Asymmetric Loss for RUL Prediction
===================================
 예측(실제보다 RUL을 크게 예측 = 고장 놓침)에 더 큰 페널티.
                                   도메인에서 안전 critical한 특성 반영.
"""
import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha_early=1.0, alpha_late=3.0):
        """
        alpha_early: 조기 예측 가중치 (RUL 과소 예측 — 일찍 정비)
        alpha_late:  지연 예측 가중치 (RUL 과대 예측 — 고장 놓침, 위험!)
        """
        super().__init__()
        self.alpha_early = alpha_early
        self.alpha_late = alpha_late

    def forward(self, pred, target):
        diff = pred - target  # 양수 = 과대 예측(위험), 음수 = 과소 예측(안전)
        weights = torch.where(diff > 0, self.alpha_late, self.alpha_early)
        return (weights * diff ** 2).mean()


class NASAScoreLoss(nn.Module):
    """NASA PHM08 공식 스코어 기반 loss (참고용, 학습에는 비대칭 MSE 추천)."""
    def forward(self, pred, target):
        diff = pred - target
        score = torch.where(diff < 0, torch.exp(-diff / 13) - 1, torch.exp(diff / 10) - 1)
        return score.mean()

# Aircraft Engine Prognostics

> NASA C-MAPSS Turbofan Engine 데이터셋 기반 RUL(잔여수명) 예측 프로젝트

## 🎯 Motivation

스마트모빌리티 전공으로 자율주행 멀티센서 데이터 처리를 학습하면서,
동일한 시계열 분석 패러다임이 항공기 예지정비(Predictive Maintenance, PdM)
분야에서 어떻게 적용되는지에 관심을 갖게 되었음.

본 프로젝트는 PHM(Prognostics and Health Management) 분야의
표준 벤치마크인 NASA C-MAPSS 데이터셋을 활용하여 RUL 예측 모델을 구현하고,
항공 안전 critical 도메인 특성을 반영한 비대칭 손실함수의 효과를 검증함.

## 📊 Dataset

**NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)**

- 100 engines, 21 sensor channels
- Single operating condition, single fault mode (HPC degradation)
- Reference: Saxena et al., PHM 2008

## 🏗️ Methodology

### Data Preprocessing
- Min-Max 정규화
- Sliding Window (30 cycles)
- Piece-wise Linear RUL labeling (cap: 125 cycles)

### Models
1. **Baseline LSTM** — 표준 LSTM 회귀
2. **CNN-LSTM** — 센서 공간 패턴 + 시계열 통합
3. **CNN-LSTM + Asymmetric Loss** — 안전 critical 손실함수 ⭐

### Loss Function
NASA 공식 평가 점수의 비대칭성을 학습 단계에 반영:
- 늦은 예측 (위험): 강한 페널티
- 빠른 예측 (안전): 약한 페널티

## 📈 Results

| Model | Loss | Test RMSE | PHM Score |
|---|---|---|---|
| LSTM | MSE | TBD | TBD |
| CNN-LSTM | MSE | TBD | TBD |
| CNN-LSTM | Asymmetric | TBD | TBD |

*(실험 진행 중)*

## 🚀 Quick Start

```bash
"""
Sliding Window Generator for LSTM Input
========================================
    → (samples, window_size, n_features) 형태로 변환
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def create_windows(df: pd.DataFrame, feature_cols: list, window_size: int = 30):
    """
    엔진별로 슬라이딩 윈도우 생성.
    
    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,) — 윈도우 마지막 시점의 RUL
    """
    X_list, y_list = [], []
    
    for uid, group in df.groupby("unit_id"):
        features = group[feature_cols].values  # (seq_len, n_features)
        rul = group["rul"].values
        
        seq_len = len(features)
        if seq_len < window_size:
            # 시퀀스가 윈도우보다 짧으면 zero-padding
            pad_len = window_size - seq_len
            padded = np.zeros((pad_len, features.shape[1]))
            features = np.vstack([padded, features])
            X_list.append(features)
            y_list.append(rul[-1])
        else:
            for i in range(window_size, seq_len + 1):
                X_list.append(features[i - window_size:i])
                y_list.append(rul[i - 1])
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def create_test_windows(df: pd.DataFrame, feature_cols: list, window_size: int = 30):
    """
    테스트: 각 엔진의 마지막 윈도우만 추출 (= 최종 시점 RUL 예측용).
    """
    X_list, y_list = [], []
    
    for uid, group in df.groupby("unit_id"):
        features = group[feature_cols].values
        rul = group["rul"].values
        
        seq_len = len(features)
        if seq_len < window_size:
            pad_len = window_size - seq_len
            padded = np.zeros((pad_len, features.shape[1]))
            features = np.vstack([padded, features])
        else:
            features = features[-window_size:]
        
        X_list.append(features)
        y_list.append(rul[-1])
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_windows(
    processed_dir: str = "data/processed",
    subset: str = "FD001",
    window_size: int = 30
):
    processed_dir = Path(processed_dir)
    
    # 메타 로드
    with open(processed_dir / f"meta_{subset}.pkl", "rb") as f:
        meta = pickle.load(f)
    feature_cols = meta["feature_cols"]
    
    # CSV 로드
    train_df = pd.read_csv(processed_dir / f"train_{subset}.csv")
    test_df = pd.read_csv(processed_dir / f"test_{subset}.csv")
    
    print(f"Building windows (size={window_size})...")
    X_train, y_train = create_windows(train_df, feature_cols, window_size)
    X_test, y_test = create_test_windows(test_df, feature_cols, window_size)
    
    # 저장
    np.save(processed_dir / f"X_train_{subset}.npy", X_train)
    np.save(processed_dir / f"y_train_{subset}.npy", y_train)
    np.save(processed_dir / f"X_test_{subset}.npy", X_test)
    np.save(processed_dir / f"y_test_{subset}.npy", y_test)
    
    print(f"✅ Windows — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    build_windows()

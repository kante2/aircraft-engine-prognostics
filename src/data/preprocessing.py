"""
NASA C-MAPSS Turbofan Engine Dataset Preprocessing
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle
from typing import Tuple, Optional

# Column definitions
INDEX_COLS = ["unit_id", "cycle"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

DROP_THRESHOLD = 1e-3


# step1 - raw data loading
# - 원본 데이터 형태 1 1 -0.0007 -0.0004 100.0 518.67 641.82 ...  / 숫자만 나열
    # train_df   ← train_FD001.txt   (학습 데이터, 고장까지 전부)
    # test_df    ← test_FD001.txt    (테스트, 중간에 잘림)
    # rul_true   ← RUL_FD001.txt     (테스트 정답, 숫자 100개)
def load_raw(data_dir: str, subset: str = "FD001"):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=ALL_COLS)#names=ALL_COLS =>  컬럼명 직접 부여
    test_df = pd.read_csv(data_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=ALL_COLS)
    rul_true = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["rul"])["rul"]
    return train_df, test_df, rul_true


# step2. — RUL 라벨 생성
def add_rul_labels(df: pd.DataFrame, max_rul: int = 125):
    df = df.copy()
    # 엔진별 최대 사이클 (= 고장 시점)
    # 엔진1: 192, 엔진2: 287, 엔진3: 145 ...
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    # 각 행에 해당 엔진의 max_cycle 값을 붙임
    df = df.merge(max_cycles, on="unit_id")
    # 엔진1, cycle=100 → rul = 192-100 = 92
    # 엔진1, cycle=192 → rul = 192-192 = 0 (고장)
    df["rul"] = df["max_cycle"] - df["cycle"]
    # 125 초과는 125로 잘라냄
    # rul=191 → 125, rul=130 → 125, rul=92 → 92 (그대로)
    df["rul"] = df["rul"].clip(upper=max_rul)
    
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_test_rul_labels(test_df: pd.DataFrame, rul_true: pd.Series):
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    test_df = test_df.merge(max_cycles, on="unit_id")
    unit_ids = test_df["unit_id"].unique()
    rul_map = dict(zip(unit_ids, rul_true.values))
    test_df["rul_final"] = test_df["unit_id"].map(rul_map)
    test_df["rul"] = test_df["rul_final"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df.drop(columns=["max_cycle", "rul_final"], inplace=True)
    return test_df

# step3 — 상수 센서 제거
# => drop_sensors = identify_constant_sensors(train_df)
def identify_constant_sensors(df: pd.DataFrame, threshold: float = DROP_THRESHOLD):
    # 센서별 표준편차 계산
        # sensor_1: std = 0.0000  ← 값이 안 변함
        # sensor_2: std = 8.3521  ← 값이 변함
    sensor_std = df[SENSOR_COLS].std()
    # 표준편차가 0.001 미만인 센서 목록 반환
        # ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    return sensor_std[sensor_std < threshold].index.tolist()


# step4 - 정규화
def normalize(train_df, test_df, feature_cols, save_path=None):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = train_df.copy()
    test_df = test_df.copy()
    #  fit_transform: train 데이터 기준으로 최소/최대값 계산 + 변환
        #  sensor_3: 최소 1580, 최대 1650
        #  → (1580-1580)/(1650-1580) = 0.0
        #  → (1620-1580)/(1650-1580) = 0.57
        #  → (1650-1580)/(1650-1580) = 1.0
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    #  transform: train에서 구한 최소/최대값을 그대로 적용
        #  ★ fit 안 함 — train 기준으로만 변환해야 데이터 누수 방지
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
    return train_df, test_df, scaler


def preprocess(data_dir="data/raw", output_dir="data/processed", subset="FD001", max_rul=125):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {subset}...")
    train_df, test_df, rul_true = load_raw(data_dir, subset)
    print(f"      Train: {len(train_df)} rows, {train_df['unit_id'].nunique()} engines")
    print(f"      Test:  {len(test_df)} rows, {test_df['unit_id'].nunique()} engines")

    print(f"[2/5] Adding RUL labels (max_rul={max_rul})...")
    train_df = add_rul_labels(train_df, max_rul=max_rul)
    test_df = add_test_rul_labels(test_df, rul_true)

    print("[3/5] Removing constant sensors...")
    drop_sensors = identify_constant_sensors(train_df)
    print(f"      Dropped: {drop_sensors}")
    feature_cols = [c for c in SENSOR_COLS + SETTING_COLS if c not in drop_sensors]
    print(f"      Remaining features: {len(feature_cols)}")

    print("[4/5] Normalizing...")
    scaler_path = output_dir / f"scaler_{subset}.pkl"
    train_df, test_df, scaler = normalize(train_df, test_df, feature_cols, save_path=scaler_path)

    # step5. — 저장
    print("[5/5] Saving...")
    train_df.to_csv(output_dir / f"train_{subset}.csv", index=False)
    test_df.to_csv(output_dir / f"test_{subset}.csv", index=False)

    meta = {
        "subset": subset, "max_rul": max_rul,
        "feature_cols": feature_cols, "drop_sensors": drop_sensors,
        "n_features": len(feature_cols),
    }
    with open(output_dir / f"meta_{subset}.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\nDone — {subset}: {len(feature_cols)} features, max_rul={max_rul}")
    return train_df, test_df


if __name__ == "__main__":
    preprocess()

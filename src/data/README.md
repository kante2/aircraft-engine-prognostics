# Dataset: NASA C-MAPSS Turbofan Engine

## Overview

NASA Ames Prognostics Center of Excellence에서 공개한
항공기 터보팬 엔진 시뮬레이션 데이터셋.
PHM (Prognostics and Health Management) 분야의 표준 벤치마크.

## Data Structure

4개 서브셋 (FD001 ~ FD004), 각 서브셋별 3개 파일:

| File | Description |
|---|---|
| `train_FD00X.txt` | 학습용: run-to-failure 데이터 (엔진별 완전 수명 시계열) |
| `test_FD00X.txt` | 테스트용: 일부 사이클까지만 (중간에 절단됨) |
| `RUL_FD00X.txt` | 테스트셋 정답 RUL 값 (엔진별 1개씩) |

## Columns (모든 .txt 파일 공통)

26 columns, space-separated:

| Col | Name | Description |
|---|---|---|
| 1 | `unit_number` | 엔진 ID (1 ~ N) |
| 2 | `time_cycles` | 비행 사이클 (시간 단위) |
| 3-5 | `op_setting_1~3` | 운전 조건 (고도, 마하수, TRA) |
| 6-26 | `sensor_1~21` | 센서 측정값 |

## Subset Description

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 (HPC + Fan) |
| FD004 | 248 | 249 | 6 | 2 |

본 프로젝트는 **FD001** 부터 시작 (가장 단순, 베이스라인 구축에 적합).

## Download

### Option A: Kaggle (추천)

```bash
# P1 새 clean O/B 전체 학습 후보

이 후보는 운영진 배포 train에서 XGBoost 1회와 event-day LightGBM 1회, 총 2회를 실제 학습한다. 과거 답안 CSV, router_anchor, GI2 행 패치, MS-TCN 저장 가중치를 입력으로 사용하지 않는다. 공식 미채점 후보이며 역사적 28.909341점을 귀속할 수 없다. 최종 제출 ZIP 및 인터넷 차단 환경의 새 학습 재현은 아직 검증하지 않았다.

## 데이터 → 코드 → 모델 → 답안

- `01_data/`: 데이터 사용 안내만. 운영진 원본 파일은 복사·재배포하지 않으며 `P1_DATA_DIR`로 읽기 전용 참조한다.
- `02_code/`: 실행 소스의 명시적 import closure, 고정 config, 작은 출처 receipt, 환경 버전.
- `03_models/`: 전체 train으로 새 학습한 `original.joblib` / `balanced.joblib` 및 `frozen_recipe.json`.
- `05_answer/P1_submission.csv`: 새 저장 모델로 생성한 169,011행 `station,year,layer,time,label` 답안. 공식 채점 전이다.
- `train_result.json`, `inference-qa.json`, `replay-qa.json`: 학습/모델봉인, schema-key-order-finite-hash, 별도 프로세스 byte equality 증거.

## 고정된 방법과 한계

2025 Q4의 이전 inner split에서 정한 O high=0.2, B_union high=0.3을 그대로 재사용한다. low ratio=0.5, close gap=0, minimum run=12. 출력은 O와 B의 후처리 양성 합집합이다. 추가 threshold 적합, 공식 점수 역산, outer label 기반 재조정은 0이다. 추가 long-flank와 binary Viterbi의 사전등록 전략은 내부에서 실패하여 사용하지 않는다.

이 방법의 같은 2025 내부 평가면에서 inner-selected control pooled F1은 0.851174였고 XGB-alone 0.843227보다 높았지만, 구간별 control 선택 및 축적된 연구가 포함된 개발 재평가다. 본 전체 학습/Q4 recipe가 공식 2026에서 같은 성능을 낸다는 보장은 없다.

중요: station×year×layer의 nominal-depth 통계는 train에서만 fit한다. 따라서 train에 없는 2026 키는 missing/unknown으로 남으며, 이 실행에서는 사후 fallback을 추가하지 않았다. Q4-inner는 seen-year이고 공식 2026은 unseen-year이므로 covariate shift 위험이 있다. test 자체의 depth 값을 학습 통계에 편입하거나 임계값을 재조정하지 않는다.

## 저장 모델 답안 재생성

Python 3.12, CPU 4 threads, BLAS 1 thread로 실행했다. 외부 데이터·모델 다운로드가 필요하지 않다. 아래는 패키지의 `02_code` 안에서 실행한다. Python 및 requirements의 wheel 준비는 인터넷 차단 전에 해야 한다.

```powershell
$env:P1_DATA_DIR = '<운영진 P1_qc_anomaly 원본 폴더>'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
python scripts/run_p1_clean_control_fulltrain_20260905_v1.py --verify --output ..
```

`--verify`는 새 모델을 학습하지 않고 저장 모델을 불러 공식 입력을 다시 추론하며 이 후보 자체의 CSV와 byte equality를 검사한다. 과거 제출 CSV와 비교하지 않는다.

## 모델까지 처음부터 재현

기존 산출물을 덮어쓰지 않도록 새 출력 폴더를 사용한다. 아래 `RETRAINED` 폴더는 기존 attempt lock이 없어야 한다.

```powershell
python scripts/run_p1_clean_control_fulltrain_20260905_v1.py --train --output ../RETRAINED
python scripts/run_p1_clean_control_fulltrain_20260905_v1.py --predict --output ../RETRAINED
python scripts/run_p1_clean_control_fulltrain_20260905_v1.py --verify --output ../RETRAINED
```

첫 명령은 train만 읽고 두 모델 및 recipe를 봉인한다. 두 번째 명령부터 공식 test와 sample의 key 열에 접근한다. sample prediction 값과 hidden truth는 읽지 않는다. 정상 학습 과정은 포함하지만 두 번째 독립 fresh 2-fit의 bitwise 재현성은 이 패키지를 준비한 현재 단계에서 아직 수행하지 않았다. 별도 프로세스의 저장 모델 inference 재현과 혼동하지 않는다.

## 제출 상태

업로드 0, commit/push 0. 파일을 준비했다는 사실은 제출 완료가 아니다. 이 답안은 OCN-01의 답안 CSV 입력에 해당하고, 최종 모델 제출용 ZIP은 별도의 검증·포장·최종 승인 절차가 필요하다. 기존 공식 패키지를 변경하거나 덮어쓰지 않았다.

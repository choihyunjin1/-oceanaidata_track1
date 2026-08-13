# Ocean AI Data Track 1 - P1

종합해양과학기지의 10분 수온 시계열에서 합성 센서 이상을 탐지하는 재현 가능한 대회 프로젝트입니다. 판정 대상은 `temp`, 참고 변수는 `psal`과 `depth`, 공식 평가지표는 행 단위 binary F1입니다. 주최측 규칙 기반 기준값은 0.548255입니다.

실험이나 제출 작업 전에는 [00_MUST_READ_FIRST.md](00_MUST_READ_FIRST.md)를 처음부터 끝까지 읽어야 합니다.

## 현재 상태

- 전용 Python 3.12.10 환경과 고정 의존성, RTX 5090 CUDA 13.0 smoke test가 준비되었습니다.
- 데이터 audit, gap-safe 피처, rolling-origin CV, LightGBM/XGBoost/CatBoost, 규칙·후처리, fold-local 합성 augmentation, TCN/Transformer, masked-reconstruction SSL, 엄격한 제출 검증 구성요소가 구현되어 있습니다.
- 주력은 배포된 전체 시계열을 사용하는 양방향 `offline` QC이며, 미래 문맥을 쓰지 않는 `causal` 모드는 실운영 가능성 비교용 ablation입니다.
- 현재 가장 강한 검증 근거는 XGBoost의 세 rolling-origin outer fold 결과입니다. 딥러닝·SSL 구성요소는 승격 전 별도 screening과 동일 검증 절차를 거칩니다.
- 원본 데이터, 캐시, 모델, OOF, 예측, 제출 CSV는 모두 Git에서 제외됩니다.
- 정확한 후보 파일에 대한 사용자 승인 없이는 대회 사이트에 업로드하지 않습니다.

학습 데이터는 776,706행(양성 32,126행, 4.1362%), test는 2026-01-01~06-30의 169,011행입니다.

## 로컬 검증 결과

7일 purge를 둔 세 rolling-origin outer fold에서, 각 outer train 내부의 과거 blocked split만으로 모델 반복 수와 후처리를 선택했습니다.

| 모델·모드 | honest outer micro F1 | test 정점·층 비중 재가중 F1 | 판정 |
|---|---:|---:|---|
| XGBoost `offline` | **0.860371** | **0.813316** | 첫 로컬 후보 |
| LightGBM `offline` | 0.816737 | 0.768804 | 비교 기준 |
| LightGBM `causal` | 0.757248 | 0.703250 | 실운영 비교 |
| CatBoost `offline` | 0.806831 | 0.757848 | 탈락 |
| LightGBM `offline` + 합성 증강 | 0.609332 | 0.553548 | 탈락 |

표의 수치는 모두 각 outer train 내부에서 선택을 끝낸 honest outer 평가입니다. train 라벨을 사용한 로컬 추정치이며 test 정답 또는 리더보드 점수가 아닙니다. 공식 0.548255도 다른 평가 집합에서 산출되었으므로 직접 차감하지 않고, 같은 로컬 fold에서 후보와 로컬 기준선을 비교합니다. 전체 판정과 후보 SHA는 `reports/P1_MODEL_SELECTION_2026-08-13.md`에 보존합니다.

고정 outer fold:

| fold | train 종료 | validation |
|---|---|---|
| 2025 Q2 | 2025-03-24 | 2025-04-01 ~ 2025-06-30 |
| 2025 Q3 | 2025-06-23 | 2025-07-01 ~ 2025-09-30 |
| 2025 Q4 | 2025-09-23 | 2025-10-01 ~ 2025-12-10 |

## 저장소 구성

| 경로 | 역할 |
|---|---|
| `src/p1_qc` | 데이터, 피처, 모델, 검증, 제출 파이프라인 |
| `configs/p1.toml` | seed, fold, 피처, 모델, 후처리 설정 |
| `tests` | 단위·통합·CUDA·누출 방지 테스트 |
| `scripts/bootstrap_env.ps1` | `.venv-p1` 생성, 고정 패키지·editable 설치, CUDA 검사 |
| `scripts/smoke_cuda.py` | CUDA capability, `sm_120`, 행렬 연산·역전파 검사 |
| `scripts/validate_submission.py` | 독립 제출 스키마·키·해시 검사 |
| `notebooks/00_p1_reconnaissance.ipynb` | 원자료 행을 노출하지 않는 재현 가능한 정찰 |
| `reports/P1_RECONNAISSANCE_2026-08-13.md` | 검증된 데이터 정찰 통계 |
| `reports/P1_IMPLEMENTATION_GUIDE.md` | 구현·검증·승격·제출 기준 |
| `reports/P1_MODEL_SELECTION_2026-08-13.md` | 모델 비교, stress, 첫 후보의 해시와 미업로드 상태 |
| `reports/ENVIRONMENT_2026-08-13.md` | Python·패키지 lock·CPU/GPU 실행 환경 |
| `reports/EXTERNAL_DATA_APPROVAL_DRAFT.md` | 외부자료와 운영 범위 서면 문의 초안 |

## 환경 구성

Windows x64용 Python 3.12.10이 설치된 프로젝트 루트에서 실행합니다. 기존 `.venv`는 건드리지 않고 `.venv-p1`을 사용합니다.

~~~powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_env.ps1
~~~

스크립트는 `requirements.txt`, 공식 CUDA 13.0 PyTorch wheel을 고정한 `requirements-dl.txt`, editable 프로젝트를 순서대로 설치한 뒤 `pip check`와 CUDA smoke test를 실행합니다. PyCharm 인터프리터는 `.venv-p1\Scripts\python.exe`를 선택합니다.

원본 폴더는 코드에 하드코딩하지 않고 환경변수 또는 CLI로 전달합니다.

~~~powershell
$env:P1_DATA_DIR = "사용자 PC의 P1_qc_anomaly 폴더"
~~~

`P1_DATA_DIR`가 없으면 필수 파일 집합을 가진 폴더를 프로젝트 아래에서 검색하며, 검색 결과가 없거나 둘 이상이면 안전하게 실패합니다.

## CLI

공개 진입점은 다음 여섯 명령입니다.

~~~text
python -m p1_qc audit|cv|train|predict|validate|reproduce --config configs/p1.toml
~~~

PowerShell 예시:

~~~powershell
# 데이터 계약과 해시 감사
.venv-p1\Scripts\python.exe -m p1_qc audit --config configs\p1.toml

# 양방향 offline 주력 CV
.venv-p1\Scripts\python.exe -m p1_qc cv --config configs\p1.toml --mode offline --backend lightgbm --bootstrap-replicates 2000

# 미래 문맥 없는 causal 비교
.venv-p1\Scripts\python.exe -m p1_qc cv --config configs\p1.toml --mode causal --backend lightgbm --bootstrap-replicates 2000

# fold-local 정상구간 합성 주입 ablation
.venv-p1\Scripts\python.exe -m p1_qc cv --config configs\p1.toml --mode offline --backend lightgbm --augment

# 승인된 selection으로 전체 train 적합
.venv-p1\Scripts\python.exe -m p1_qc train --config configs\p1.toml --selection <selection.json>

# test 추론과 엄격 검증
.venv-p1\Scripts\python.exe -m p1_qc predict --config configs\p1.toml --model <model.joblib> --output <candidate.csv>
.venv-p1\Scripts\python.exe -m p1_qc validate --config configs\p1.toml <candidate.csv>

# 저장 가중치에서 행 단위 동일성 재현
.venv-p1\Scripts\python.exe -m p1_qc reproduce --config configs\p1.toml --model <model.joblib> --output <reproduced.csv> --expected <candidate.csv>
~~~

각 run은 ignored `artifacts/runs/<run_id>` 아래에 설정, Git 상태, 입력 SHA-256, 환경·GPU 정보, 피처·분할 해시, 지표, 선택 설정과 산출물 해시를 기록합니다.

## 정찰 노트북

~~~powershell
.venv-p1\Scripts\python.exe -m jupyter execute --inplace --timeout=600 --kernel_name=python3 notebooks\00_p1_reconnaissance.ipynb
~~~

노트북은 원자료 행을 출력하지 않고 행 수, 결측률, 키 무결성, 시간 간격, 라벨과 연속 구간의 집계 통계만 표시합니다.

## 고정 안전 원칙

- 원본 ZIP과 추출 파일은 읽기 전용이며 재배포하지 않습니다.
- 행 단위 무작위 분할, test 양성률 4% 강제, 결측을 정답처럼 사용하는 모델을 금지합니다.
- centered 피처는 동일 연속 segment 안에서만 계산하며 fold의 purge가 최대 의존 범위를 덮어야 합니다.
- causal 결과는 offline 결과와 분리해 보고합니다.
- 외부 관측값은 운영진 서면 승인 전 0건 사용합니다. 2024~2026 KORS/KHOA 원자료·실시간값은 승인 여부와 관계없이 제외합니다.
- 일일 1회 제출 기회를 보호하기 위해 로컬 validator와 재현 검사를 통과한 정확한 파일만 사용자에게 제시합니다.
- 최종 모델 지정 뒤 예측 업로드가 잠길 수 있으므로 별도 경고와 사용자 재확인 없이 최종 모델을 지정하지 않습니다.

## 공식 일정

| 출처 | 단계 | 일정 |
|---|---|---|
| 최신 대회 UI | 문제 공개 | 2026-08-13 |
| 최신 대회 UI | 제출 마감·최종 모델 | 2026-09-07(정확한 시각 확인 필요) |
| 초기 공고 PDF | 온라인 해커톤 | 2026-08-10 ~ 2026-09-04 |
| 초기 공고 PDF | 코드·데이터셋 zip 마감 | 2026-09-04 10:00 |
| 초기 공고 PDF | 본선 진출자 발표 | 2026-09-11 |
| 초기 공고 PDF | 본선 자료 제출 | 2026-09-29 |
| 초기 공고 PDF | 본선 심사 | 2026-10-08 |

최신 UI 일정이 현재 작업 기준입니다. 초기 PDF 일정은 이전 일정과 재현성 요구의 근거로만 보존하며, 9월 7일 정확한 마감 시각과 최종 모델 지정·zip 업로드 관계는 최종 행동 전에 다시 확인합니다.

## 공식 링크

- [KIMST 대회 공고](https://www.kimst.re.kr/u/news/notice_01/board.do?bno=153421765145711&searchDiv=&searchKeyword=&type=view)
- [경진대회 홈페이지](https://www.oceanaidata.org)
- [GitHub 백업 저장소](https://github.com/choihyunjin1/-oceanaidata_track1)

# P1 데이터 로딩 벤치마크

> 측정일: 2026-08-13 KST
> 대상: 배포된 `train.csv`, `test.csv`의 반복 로딩과 로컬 ZSTD Parquet 캐시
> 원본 정책: 읽기 전용, 원자료 행 출력 0건, 외부 다운로드 0건
> 상세 실행 결과: ignored `artifacts/cache/io_benchmark/benchmark_results.json`

## tl;dr

- 현재 pandas 파이프라인과 실제 호환되는 기본 원자료 로더로는
  `pandas.read_csv(engine="pyarrow", dtype_backend="pyarrow")`가 가장 낫다. train/test
  합계 median은 **0.167초**였다. 이 입력으로 train 776,706행과 test 169,011행 모두
  offline feature 80개 생성이 완료되었고 train 양성 32,126건도 보존되었다.
- ZSTD Parquet은 저장공간을 CSV의 train **12.31%**, test **16.94%**로 줄였지만,
  pipeline-compatible pandas 기준 train/test 합계 로드는 0.130초로 Arrow CSV보다
  0.037초만 빨랐다. 빌드 0.528초를 회수하려면 같은 원본을 약 **15회** 다시 읽어야
  한다. test만 보면 Arrow CSV가 Parquet보다 빨라 break-even이 없다.
- 따라서 **원본 최초·변경 시에는 기존 strict audit와 SHA-256을 수행하고, 동일 해시의
  반복 실행에는 Arrow CSV를 사용**하는 것을 기본 경로로 권한다. 원자료 Parquet은
  저장공간 절약, Polars 실험, 아주 잦은 반복용 선택 캐시로 둔다.
- 실제 병목은 raw I/O가 아니다. 최근 완료 CV 4회의 artifact span은 605.1~725.4초,
  median 683.7초였다. 현재 strict ingestion 8.795초는 약 1.29%, cold offline feature
  생성 20.291초는 약 2.97%이며, 나머지 약 **95.75%**는 반복 모델 학습·추론·후처리·
  bootstrap 영역이다. feature 캐시가 적중하면 학습 영역 비중은 더 커진다.
- offline feature 생성의 train peak RSS 증가량 median은 **1,296 MiB**였다. 속도보다
  메모리 여유와 feature 캐시 무효화 계약이 운영상 더 중요한 위험이다.

## 1. 데이터와 측정 단위

| 데이터 | 행 | 열 | CSV 크기 | ZSTD Parquet 크기 | Parquet/CSV |
|---|---:|---:|---:|---:|---:|
| train | 776,706 | 9 | 48.241 MiB | 5.939 MiB | 12.31% |
| test | 169,011 | 7 | 9.874 MiB | 1.673 MiB | 16.94% |

관측 grain은 `station, year, layer, time` 한 행이다. 모든 로더에서 다음 집계 계약이
일치해야 결과를 성능표에 포함했다.

- 행 수, 열 수, 열 순서
- 열별 `null 또는 빈 문자열` 정규화 개수
- train `label=1` 개수
- station 및 layer 고유 개수
- Parquet의 `time`을 timestamp로 바꾸지 않고 원래 `+09:00` 문자열 키로 보존

최종 계약은 train 양성 32,126건, station 3개, layer 8개를 일관되게 확인했다. test의
psal 결측 798건과 depth 결측 16,368건도 12개 로더 모두에서 일치했다. 개별 관측값은
로그나 보고서에 출력하지 않았다.

## 2. 방법

재현 명령:

```powershell
$env:P1_DATA_DIR = "로컬 P1_qc_anomaly 폴더"
.venv-p1\Scripts\python.exe scripts\benchmark_data_io.py --repeats 3
.venv-p1\Scripts\python.exe -m pytest -q tests\test_benchmark_data_io.py
```

측정 설계:

1. 라이브러리 import 후, 각 로드를 새 Python 프로세스에서 실행했다.
2. train/test 및 메서드 순서를 seed 20260813으로 라운드마다 섞었다.
3. 모든 메서드는 3회 측정해 median과 선형 보간 p95를 계산했다.
4. Windows OS 파일 캐시는 강제로 제거하지 않았다. 측정 전에 파일을 순차 읽어
   warm-cache 반복 개발 환경을 명시적으로 만들었다.
5. wall time은 실제 materialization까지 포함한다. `scan_csv`와 `scan_parquet`은 lazy
   plan 생성만 잰 값이 아니라 `collect(engine="streaming")` 완료까지의 시간이다.
6. RSS는 2 ms 간격으로 별도 thread가 관측했다. 각 dataframe/table의 자체 메모리
   추정치도 기록했다.
7. ZSTD level 3 Parquet 빌드도 새 프로세스에서 3회 재생성해 median/p95를 냈다.
8. source와 cache의 SHA-256, 스키마·품질 계약 및 패키지 버전을 ignored JSON에 보존했다.

환경은 Python 3.12.10, pandas 3.0.1, PyArrow 25.0.1, Polars 1.43.2,
psutil 7.2.2였다.

## 3. 실제 로더 결과

단위는 초와 MiB다. `peak`는 로드 직전 RSS 대비 증가량 median이고 `object`는 각
프레임워크 자체 추정 dataframe/table 크기 median이다.

### 3-1. CSV

| 메서드 | train median | train p95 | train peak | train object | test median | test p95 | test peak | test object |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 현재 `p1_qc` strict audit+CSV | 7.456 | 8.518 | 305.4 | 75.8 | 1.339 | 1.660 | 73.0 | 13.9 |
| pandas 기본 C engine | 0.965 | 1.075 | 221.8 | 75.8 | 0.183 | 0.257 | 42.6 | 13.9 |
| pandas 명시 dtype/usecols | 0.849 | 0.852 | 253.4 | 43.2 | 0.154 | 0.163 | 54.0 | 7.9 |
| pandas PyArrow engine/backend | **0.133** | 0.165 | 155.3 | 51.5 | **0.033** | 0.125 | 32.3 | 9.2 |
| PyArrow CSV table | 0.087 | 0.109 | 107.7 | 43.4 | 0.026 | 0.031 | 36.3 | 8.6 |
| Polars eager | **0.066** | 0.074 | 144.1 | 58.1 | **0.018** | 0.032 | 32.7 | 10.8 |
| Polars lazy+collect | 0.103 | 0.136 | 141.8 | 58.1 | 0.023 | 0.025 | 35.5 | 10.8 |

`p1_qc` strict 경로는 단순 CSV 파싱뿐 아니라 SHA-256과 전체 audit를 포함하므로 순수
로더와 같은 의미의 수치는 아니다. 다만 현재 CLI 사용자가 체감하는 입력 단계의
비용을 나타내는 비교 기준이다.

### 3-2. 생성된 ZSTD Parquet

| 메서드 | train median | train p95 | train peak | train object | test median | test p95 | test peak | test object |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pandas/NumPy backend | 0.082 | 0.098 | 117.1 | 52.1 | 0.050 | 0.069 | 37.5 | 9.8 |
| pandas/Arrow backend | 0.096 | 0.097 | 108.1 | 43.5 | 0.048 | 0.052 | 34.3 | 8.6 |
| PyArrow `read_table` | 0.469 | 0.674 | 150.0 | 43.5 | 0.431 | 0.462 | 74.1 | 8.6 |
| Polars eager | **0.044** | 0.055 | 77.5 | 34.4 | 0.011 | 0.012 | 17.2 | 7.3 |
| Polars lazy+collect | 0.047 | 0.073 | 77.8 | 34.4 | **0.008** | 0.019 | 17.0 | 7.3 |

격리된 일회성 CLI 프로세스에서는 `pyarrow.parquet.read_table`의 최초 dataset 초기화가
측정 구간에 들어가 CSV보다 느렸다. 이미 초기화된 장기 실행 프로세스의 반복 read를
대표하지 않으므로 PyArrow 자체의 일반적인 Parquet 성능으로 확대 해석하면 안 된다.

## 4. 캐시 빌드와 break-even

| 데이터 | build median | build p95 | build peak RSS | pandas 호환 CSV | pandas 호환 Parquet | load speedup | break-even |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.440초 | 0.476초 | 142.4 MiB | 0.133초 | 0.082초 | 1.62x | 9회 |
| test | 0.089초 | 0.103초 | 42.8 MiB | 0.033초 | 0.048초 | 0.70x | 없음 |
| 합계 | 0.528초 | — | — | 0.167초 | 0.130초 | 1.28x | 15회 |

break-even은 `ceil(build time / (CSV median - Parquet median))`으로 계산했다. 파일 크기
절감은 크지만, 이미 매우 빠른 pandas PyArrow CSV와 비교하면 wall-time 이익은 작고
test에서는 오히려 Parquet가 느렸다. “한 번 빌드하면 다음 read부터 무조건 이익”이라는
결론은 pandas 기본/typed CSV만 비교할 때 생기는 잘못된 결론이다.

Polars로 전체 feature 파이프라인을 옮기면 Parquet의 로드와 메모리가 가장 작지만,
현재 pandas 기반 gap segmentation·rolling·층간 정렬을 다시 구현하고 누출 동등성을
검증해야 한다. 로드 수백 ms 절감을 위해 지금 이 이식 위험을 감수할 근거는 없다.

## 5. 전체 파이프라인 병목 비중

offline feature 생성 자체를 원자료 로드와 분리해 3회 측정했다.

| 단계 | train median / p95 | test median / p95 | peak RSS median | 결과 크기 |
|---|---:|---:|---:|---:|
| offline feature 80개 | 16.350 / 17.838초 | 3.941 / 4.612초 | train 1,296.3 MiB, test 144.9 MiB | train 260.3 MiB, test 56.6 MiB |

현재 run recorder에는 단계별 timer가 없으므로 모델 영역은 최신 완료 CV artifact의 생성
시각부터 마지막 결과 파일 시각까지를 보조 근거로 썼다. 최근 4개 완료 CV span은
605.1, 665.8, 701.6, 725.4초였고 median은 683.7초다. artifact 후편집이나 동시 작업의
영향을 받을 수 있어 정밀 profiler가 아니라 병목 규모 판정용이다.

| 비교 구성 | train+test 시간 | CV median 대비 |
|---|---:|---:|
| 현재 strict ingestion | 8.795초 | 1.286% |
| 권장 pandas Arrow CSV 파싱만 | 0.167초 | 0.024% |
| cold offline feature 생성 | 20.291초 | 2.968% |
| strict ingestion+feature를 제외한 나머지 | 약 654.6초 | 약 95.746% |

따라서 입력 최적화만으로 CV 총 시간을 크게 줄일 수 없다. 기존 feature Parquet 캐시가
적중하면 20.291초 feature 생성도 생략되어 반복 모델 fit, postprocess 탐색과 bootstrap의
비중은 더 커진다. 다음 성능 엔지니어링 우선순위는 fold별 중복 matrix 변환 제거,
LightGBM/XGBoost/CatBoost fit 병렬성 검토, postprocess grid의 벡터화, bootstrap 결과
캐시다.

## 6. 권장 기본 경로

1. 원본 파일을 처음 보거나 size/mtime/source SHA가 바뀐 경우에는 기존
   `load_dataset(..., audit=True, strict=True)`를 반드시 실행한다.
2. audit 결과와 source SHA-256을 작은 manifest로 캐시한다. 동일 SHA의 반복 실행에서는
   `pandas.read_csv(engine="pyarrow", dtype_backend="pyarrow")`를 사용한다.
3. 이 Arrow-backed pandas 입력은 train/test 전체 offline feature 생성 계약을 통과했다.
   적용 전 단위시험에는 pandas 기본 입력과 feature 행·열·결측 mask의 동등성 검사를
   추가한다.
4. 현재의 **feature Parquet cache는 유지**한다. raw Parquet를 기본으로 강제하지 않는다.
5. raw ZSTD Parquet는 동일 source SHA에서 15회 이상 반복하거나, 작은 저장공간이
   중요하거나, Polars 실험을 수행할 때만 생성한다. cache manifest에 source SHA,
   schema version, compression, builder version을 기록한다.
6. 캐시 파일명과 cache root는 ASCII로 유지하고 `artifacts/cache` 밖으로 복사하거나 Git에
   추가하지 않는다.

## 7. 한계와 신뢰도

- **warm-cache 전용:** Windows Standby List를 비우지 않았다. cold disk, 다른 SSD,
  네트워크 드라이브 성능을 추정할 수 없다.
- **p95 표본 제한:** n=3의 p95는 대략적인 최대값에 가까운 기술통계일 뿐 SLA tail
  estimate가 아니다. median 판단은 신뢰도가 높고 p95 판단은 낮다.
- **동시 부하:** 워크스테이션에서 다른 개발 작업이 병행될 수 있다. 각 방법을 교차
  순서·독립 프로세스로 실행해 편향을 줄였지만 제거하지는 못했다.
- **RSS 하한:** 2 ms 사이에 발생했다 사라진 native allocation은 놓칠 수 있다.
- **메모리 추정 차이:** pandas `memory_usage(deep=True)`, Arrow `nbytes`, Polars
  `estimated_size`는 정의가 동일하지 않다. 프레임워크 간 object MiB는 방향성 비교다.
- **모델 단계 비중:** CV artifact span 기반 근사다. 정확한 단계 비중을 위해서는
  run recorder에 `load/audit`, `feature`, `encode`, `fit`, `predict`, `postprocess`,
  `bootstrap` monotonic timer를 추가해야 한다.

## 8. 자동 회귀시험

`tests/test_benchmark_data_io.py`는 합성 train/test에서 다음을 고정한다.

- 11개 직접 loader와 ZSTD cache의 행·열·열 순서 동등성
- 결측/빈 문자열 정규화와 label 양성 수 동등성
- Parquet time 문자열 키 보존
- feature profile의 Parquet 비의존성
- p95와 break-even 계산
- schema/shape mismatch 실패
- cache output이 ignored `artifacts` 밖으로 나가지 못하는 경로 게이트

실측 결과와 캐시는 재배포 금지 원본의 파생 로컬 artifact이므로 Git 대상이 아니다.

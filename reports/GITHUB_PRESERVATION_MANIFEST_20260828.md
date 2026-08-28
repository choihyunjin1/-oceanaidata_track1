# GitHub 보존 manifest — 2026-08-28

## 포함 범위

- 루트 운영 문서: `README.md`, `AGENTS.md`, `00/01/02_MUST_READ_FIRST.md`
- `configs/`: 고정 실험 계약, 외부 데이터 정책, 실행·복구 기록
- `docs/`: 재현 및 운영 문서
- `scripts/`: 빌더, runner, QA, 분석 및 보고서 생성기
- `src/`: P1/P2/P3 모델·검증·외부 데이터 코드
- `tests/`: 단위·계약·회귀·독립 QA 테스트
- `reports/`: 연구 결론, 공식점수 원장, QA 결과, 시각 보고서
- `requirements-era5.txt`, `requirements-external.txt`, `requirements-p3-tsfm.txt`

## 2026-08-28 공식 검증 스냅샷

| 문제 | 보존된 Public 최고 | 점수 | 직전 최고 대비 |
|---|---:|---:|---:|
| P1 | F1 0.833548 | 28.909341 | +0.007978점 |
| P2 | RMSE 0.430250 °C | 27.934759 | +0.012572점 |
| P3 | RMSE 0.575262 m | 24.203126 | +0.136958점 |

- 팀 `분당독고다이`: 총점 **81.047226**, Public **4위**.
- 세부 제출별 점수·비교는 `reports/deadline_submission_results_20260828_v1/official-results.md`에 기록했다.
- 점수는 Public 분할 기준이며, Private 순위나 최종 일반화 성능을 보장하지 않는다.

## 제외 범위

| 경로/유형 | 제외 이유 |
|---|---|
| `/external_data/` | ERA5·외부 원본/파생 데이터, 재배포·용량 제한 |
| `/data/`, `/datasets/`, `/데이터셋 원본/` | 대회 원본 데이터 |
| `/output/`, `/outputs/`, `/submissions/`, `/artifacts/` | 제출 CSV, 모델, 대용량 실행 산출물 |
| `/tmp/` | browser profile, cache, 임시 모델 |
| `/reports/**/xlsx_work/`, `/reports/**/node_modules/` | 보고서 생성용 제3자 runtime |
| `/reports/**/rendered*/` | Markdown/DOCX에서 재생성 가능한 중복 PDF·페이지 PNG |
| `.venv*`, `__pycache__`, `*.pyc` | 로컬 환경/캐시 |
| `.env*` | credential 및 로컬 비밀정보 |

## 제외 데이터의 재현 식별자

- P3 ERA5 combined parquet: `5106c4ee35c7d434dcea13d1b436691eea9b05ef9f8c59fdd900d4c19bad9ac1`
- P3 ERA5 canonical manifest: `72c1b49791a1c73be34f8cf9c78430e074354dfad5a83f0d891edf46e20752b2`
- P3 one-shot attempt lock: `da35167d20ea2d2aa62a13bb885b9966b4c655399fac22495eeb74c293e811f7`
- P3 최종 제출 CSV: `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa`
- P2 OAS20 제출 CSV: `f46dec7944fe4565307b0242fdab5772a684027f1a42a62404d6e01ba13e0ef7`

## 보존 원칙

원본·제출 파일을 Git에 복제하지 않고, 코드·고정 계약·검증 결과·해시를 통해 연구 계보를 재현한다. force push, reset, rebase 또는 기존 이력 폐기는 허용하지 않는다.

오늘 생성된 일회성 패키지 빌더는 개인 절대경로를 제거하고 `--submission-archive`, `--p2-data-dir`, `--output-dir` 인자로 이식 가능하게 보존한다. 실행 영수증과 과거 보고서 안의 로컬 경로 문자열은 당시 파일 계보를 설명하는 기록일 뿐이며, 해당 데이터 파일 자체는 Git에 포함하지 않는다.

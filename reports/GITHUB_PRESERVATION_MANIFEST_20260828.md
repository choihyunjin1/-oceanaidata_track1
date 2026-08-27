# GitHub 보존 manifest — 2026-08-28

## 포함 범위

- 루트 운영 문서: `README.md`, `AGENTS.md`, `00/01/02_MUST_READ_FIRST.md`
- `configs/`: 고정 실험 계약, 외부 데이터 정책, 실행·복구 기록
- `docs/`: 재현 및 운영 문서
- `scripts/`: 빌더, runner, QA, 분석 및 보고서 생성기
- `src/`: P1/P2/P3 모델·검증·외부 데이터 코드
- `tests/`: 단위·계약·회귀·독립 QA 테스트
- `reports/`: 연구 결론, 공식점수 원장, QA 결과, 시각 보고서
- `requirements-era5.txt`, `requirements-external.txt`

## 제외 범위

| 경로/유형 | 제외 이유 |
|---|---|
| `/external_data/` | ERA5·외부 원본/파생 데이터, 재배포·용량 제한 |
| `/data/`, `/datasets/`, `/데이터셋 원본/` | 대회 원본 데이터 |
| `/output/`, `/outputs/`, `/submissions/`, `/artifacts/` | 제출 CSV, 모델, 대용량 실행 산출물 |
| `/tmp/` | browser profile, cache, 임시 모델 |
| `/reports/**/xlsx_work/`, `/reports/**/node_modules/` | 보고서 생성용 제3자 runtime |
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

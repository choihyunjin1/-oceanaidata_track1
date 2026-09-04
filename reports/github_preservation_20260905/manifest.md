# GitHub 보존 manifest — 2026-09-05

## 결론

이번 커밋은 2026-08-30~09-05 사이의 P1/P2/P3 연구·실행·QA 기록과 공식 최종 제출 재현 도구를 한 번에 보존한다. stage 직전 집계는 1,943개 파일, 약 11.9 MB의 신규 text/notebook 자료와 9개 기존 tracked 수정이다. 운영진 원본 데이터, 제출 CSV, 예측 NPZ/parquet, checkpoint, cache, credential은 포함하지 않는다.

## 포함 범위

| 최상위 경로 | 파일 수 | 내용 |
|---|---:|---|
| `configs/` | 267 | preregistration, experiment, compliance contracts |
| `reports/` | 952 | 결과, gate, 독립 QA, 실패·기술 blocker 기록 |
| `scripts/` | 429 | 실행, materialization, QA, 최종 package builder |
| `tests/` | 265 | focused regression and contract tests |
| `src/` | 19 | 재사용 가능한 P1/P2/P3 모델·특징 모듈 |
| `notebooks/` | 3 | P1/P2/P3 독립 최종 제출 notebook template |
| root/docs | 8 | 최상위 데이터 정책, 문제별 운영 문서, 제출 안내 |

확장자 집계는 JSON 808, Python 713, Markdown 417, notebook 3, TOML 1, TXT 1이다. stage 대상에서 원본/제출/모델 산출물 확장자 CSV, parquet, NPZ, PT, CBM은 0개다.

## 제외 범위

- `artifacts/`, `output/`, `outputs/`, `submissions/`: 실행 산출물, 제출 후보, 모델, cache
- 운영진 배포 원본: P1/P2/P3 train, test, sample, baseline, context
- P1 checkpoint 및 prediction NPZ, P3 CatBoost model bundle
- `.env`, API token, credential, 개인 secret
- Python cache, pytest cache, 임시 파일

로컬 공식 최종 패키지는 `artifacts/official_final_submission_20260905/`에 유지하며 Git에는 들어가지 않는다.

## 안전 검사

- 실제 credential 형식 6종에 대한 repository scan: hit 0
- stage 대상 forbidden data/model filename 및 확장자: 0
- focused policy/final-package pytest: 16 passed
- 새 final-package 코드 Ruff: PASS
- `git diff --check`: whitespace error 0 (Windows line-ending warning만 존재)
- 로컬 package independent QA: PASS

## 최종 패키지 결과

- P1: 169,011행, SHA `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687`
- P2: 26,061행, SHA `331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9`
- P3: 1,200행, SHA `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa`

세 notebook은 각각 별도 kernel에서 code cell 4개를 모두 실행했고 error output은 0개였다. 홈페이지 최종 제출은 이 커밋에서 수행하지 않는다.

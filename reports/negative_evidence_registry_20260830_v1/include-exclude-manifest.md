# 2026-08-30 보존 범위

## 포함

- 2026-08-29 robust-repair P1/P2/P3의 작은 config, 실행 코드, 단위 테스트와 집계 보고서
- P1 Group-DRO 목적함수와 P3 valid-only CatBoost HPO 지원 코드
- P2 train-only copula support audit의 집계 JSON
- 마감 정보가치 후보를 재현하는 portable builder 코드
- 과거 음성 증거를 exact family 단위로 합친 본 원장

## 제외

- 모든 공식 test/sample/submission CSV와 다운로드·업로드용 CSV
- raw/source data와 행별 prediction, NPZ, parquet
- 모델 checkpoint와 cache
- 실행 log, stderr 원문, attempt lock
- credential, token, `.env`
- 개인 PC 절대경로

P3 기술 실패의 원문 traceback과 selection artifact는 Git에 넣지 않고, 작은 hash·원인·fit count만 `parallel_robust_repair_cycle_20260829_v2` 보고서에 보존한다.

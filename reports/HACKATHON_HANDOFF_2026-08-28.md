# 해양 AI 해커톤 작업 인계 및 보존 보고서

작성 기준: 2026-08-28 00:00 KST 전후  
팀: 분당독고다이  
저장소: `choihyunjin1/-oceanaidata_track1`  
브랜치: `codex/p1-qc`

## 결론

- P1·P2·P3 연구 코드, 고정 실험 설정, 테스트, 연구 보고서와 재현 지침을 Git 보존 대상으로 정리했다.
- 2026-08-27 마지막 확인 공식 점수는 P1 `28.901363`, P2 `27.264587`, P3 `24.066168`, 합계 `80.232118`이다.
- P2는 계절 국소 OAS T/S 프로파일을 20% 결합한 후보가 기존 `26.611283`에서 `+0.653304점` 개선했다.
- P3의 Public long-lead 이차축은 `alpha=-10.217432`에서 `24.066168`로 사실상 수렴했다. 같은 축의 추가 미세조정은 가치가 거의 없다.
- P3 ERA5 원본 다운로드와 데이터 preflight는 모두 완료됐지만, 고정 one-shot 실행은 CatBoost가 없는 다운로드 전용 환경에서 시작되어 모델 fit 전에 종료됐다. attempt lock은 소비됐으므로 결과 기반 재실행이나 임의 lock 삭제는 하지 않았다.

## 공식 제출 기록

| 문제 | 후보 | Public RMSE | 공식 점수 | 해석 |
|---|---|---:|---:|---|
| P1 | 2026-08-27 최고 제출 | - | 28.901363 | 당일 리더보드 최고 |
| P2 | 기존 U | 0.535727 | 26.611283 | 비교 기준 |
| P2 | SEASONAL-OAS-TS-10 | 0.507628 | 26.963865 | +0.352582점 |
| P2 | SEASONAL-OAS-TS-20 | 0.483661 | 27.264587 | 기존 대비 +0.653304점 |
| P3 | long alpha*=-10.235445 | 0.583892 | 24.066167 | 기존 Public 이차축 최고 |
| P3 | long alpha=-12 | 0.584611 | 24.054757 | 곡선 반대편 검증 |
| P3 | refined alpha=-10.217432 | 0.583892 | 24.066168 | +0.000001점, 축 수렴 확인 |

Private 지표는 대회 종료 후 공개되므로 위 수치는 Public에 한정한다.

## P3 ERA5 최종 데이터 상태

- raw monthly NetCDF: `363/363`
- `.partial`: `0`
- derived monthly files: `363/363`
- combined rows: `262,917`
- 기간: `2014-01-01T00:00:00Z`부터 `2023-12-31T14:00:00Z`
- 공통 past-only feature: `286`
- combined parquet SHA-256: `5106c4ee35c7d434dcea13d1b436691eea9b05ef9f8c59fdd900d4c19bad9ac1`
- canonical manifest SHA-256: `72c1b49791a1c73be34f8cf9c78430e074354dfad5a83f0d891edf46e20752b2`
- check-only preflight: PASS (`source_quarantine_ready=true`, `source_preflight.accepted=true`)

오래된 빈 canonical manifest는 고정 collision-recovery v2의 exact SHA·process-exit·unused-attempt guard를 통과한 뒤 recoverable backup으로 이동했다. 완료된 363개 derived 파일을 network-free `--stage combine`으로 검증해 새 canonical manifest를 만들었다.

## P3 ERA5 one-shot 종료 원인

`scripts/run_p3_era5_context_transfer_v1.py --execute`는 2026-08-27 23:59:36 KST에 정확히 한 번 호출됐다. 실행은 source case 구성 뒤 첫 CatBoost import에서 다음 오류로 끝났다.

```text
ModuleNotFoundError: No module named 'catboost'
```

원인은 ERA5 다운로드 전용 `.venv-era5`가 의도적으로 ML/DL stack을 제외하는데도 그 환경으로 과학 runner를 실행한 것이다. `.venv-p1`에는 고정 버전 `catboost==1.2.10`이 존재한다. 실패 시점에는 모델 fit, prediction, source gate, 세 local window, blind prediction seal 모두 생성되지 않았다.

- attempt lock SHA-256: `da35167d20ea2d2aa62a13bb885b9966b4c655399fac22495eeb74c293e811f7`
- output directory: 존재하지만 파일 `0개`
- `result.json`: 없음
- blind prediction seal: 없음
- 상태: `TERMINAL_DEPENDENCY_FAILURE_AFTER_ONE_SHOT_LOCK`

고정 one-shot 규칙 때문에 lock을 삭제하거나 결과를 본 뒤 다시 실행하지 않았다. 향후 같은 설계의 새 preregistered attempt를 만들 경우 과학 runner의 시작 전 preflight에 `catboost`, `sklearn`, `numpy`, `pandas`, `pyarrow` import/version 검사를 반드시 포함하고, 다운로드 환경과 모델 환경을 명시적으로 분리해야 한다.

## 다음 연구 우선순위

1. P3: Public alpha축은 종료하고 ERA5 또는 다른 causal forcing을 쓰는 새 오류공간을 별도 preregistration으로 검증한다.
2. P2: OAS 20%의 공식 상승을 출발점으로 blend strength와 계절·수심별 gate를 독립 local surface에서 검증한다.
3. P1: checkpoint 최고점 보존과 long-event 구조를 유지하되 공식 점수 `+3` 목표는 새로운 segment scorer/backbone에서 찾아야 한다.
4. local 지표와 공식 점수의 변환은 문제별로 따로 기록한다. 단위 RMSE 변화와 33점 공식 점수를 혼용하지 않는다.

## 보존 경계

Git에는 코드, 설정, 테스트, 보고서, 작은 JSON/Markdown 지표와 재현 지침만 포함한다. 다음은 의도적으로 제외한다.

- 대회 원본 데이터 및 공식 test/sample/submission
- ERA5 raw/derived 파일
- 제출용 CSV와 Downloads 폴더
- 모델 checkpoint, browser profile, cache, 임시파일
- credential, token, `.env`
- 보고서 생성용 `node_modules`/`xlsx_work`

제외된 과학 데이터는 위 SHA-256과 manifest로 식별하며 GitHub에 재배포하지 않는다.

# P1·P2·P3 승인 실행 최종 보고서

작성 기준: 2026-08-28 KST
팀: 분당독고다이
상태: 고정 실험 실행·독립 QA 완료, 신규 공식 업로드 0회

## 결론

- **P1은 `NO_GO_SYNTHETIC_FIDELITY`다.** exact degradation-mask Transformer는 noise를 잘 찾았지만 offset·drift 경계를 정확히 자르지 못했다. 실제 gate나 공식 입력은 열지 않았고 CSV도 만들지 않았다.
- **P2는 `NO_GO_KEEP_OAS40_NO_CSV_NO_RESEARCH_LOOP`다.** 세 historical fold 모두 public-feature shift discriminator AUC가 1.000이라 안전한 importance weighting 조건이 성립하지 않았다. 안전 veto가 correction을 0행으로 만들었으므로 현 OAS40 champion을 유지한다.
- **P3만 `READY_NOT_UPLOADED`로 승격됐다.** 현 champion 계보에서 다시 계산한 Hs-squared residual은 historical OOF에서 RMSE를 0.014528 m 줄였고 3/3 folds, 3/3 stations, 18h와 24h가 모두 개선됐다. 이를 사전 고정된 286개 past-only feature ERA5 전이모델로 공식 무정답 입력에 배포해 1,200행 후보를 만들었다.
- P3 후보는 **제출 가치가 있다.** 단, 예상 Public 이득 중심값 약 +0.231점과 단순 CI 운반 +0.149~+0.319점은 historical 효과가 그대로 운반된다는 시나리오일 뿐 보장이 아니다. Public/Private 분할과 과거 방향 반전 때문에 실제 공식 검증이 필요하다.

## P1 결과

실험: `p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1`

- fit 1회, 16 epochs, 544 optimizer steps, best epoch 10
- best synthetic macro raw F1: 0.545001
- noise: F1 0.980603, boundary MAE 2.764 rows — 통과
- offset: F1 0.805349, boundary MAE 17.708 rows — 경계 gate 실패
- drift: F1 0.757060, boundary MAE 38.333 rows — F1·경계 gate 실패
- calibration/qualification/Q2/Q3/Q4/공식 입력 접근: 0
- CSV·업로드·anchor 삭제·point adjustment·smoothing·truth fill: 0
- 독립 QA: PASS

판정: exact-mask supervision 자체는 noise 계열에 유효하지만, 현재 형태로는 느린 offset/drift의 시작·끝 경계를 복원하지 못한다. 이 구조는 현재 명세 그대로 재실행하지 않는다.

## P2 결과

실험: `p2_oas40_target_weighted_nonlinear_thermocline_residual_20260828_v1`

- 3 folds, 69,850 historical rows
- 세 fold의 public-feature shift discriminator AUC: 모두 1.000
- correction enabled rows: 0
- OAS40 대비 RMSE 변화: 0.000000 C
- fallback exact no-op maximum absolute difference: 0
- parameter search·공식 입력·hidden answer·CSV·upload: 0
- 독립 QA: PASS

판정: source와 query support가 완전히 분리돼 이번 importance-weighted thermocline residual은 검증 가능한 correction을 만들지 못했다. 현 공식 OAS40 champion을 유지하고 이 계열은 닫는다.

## P3 historical champion-matched replay

실험: `p3_champion_lineage_matched_energy_residual_replay_20260828_v1`

사전 고정 규칙은 18h·24h에서만 다음을 적용하고 3h·6h·9h·12h는 champion을 bit-exact 유지한다.

`Hs_candidate = sqrt(max(Hs_champion^2 + 0.25 * (Hs_transfer^2 - Hs_champion^2), 0))`

- 181 cases / 1,086 rows
- champion RMSE: 0.807598 m
- candidate RMSE: 0.793070 m
- delta: -0.014528 m
- folds improved: 3/3
- stations improved: 3/3
- lead 18h delta: -0.034163 m
- lead 24h delta: -0.042431 m
- paired-case bootstrap CI90: [-0.020082, -0.009369] m
- bootstrap P(candidate improves): 1.000
- new fit/search/official rows/CSV/upload: 0/0/0/0/0
- 독립 QA: PASS

현 Public 구간 환산 기울기 15.872414 points/m를 단순 적용하면 중심 점수 변화는 +0.2306점이고 CI 운반 범위는 +0.1487~+0.3187점이다. 이는 공식 예측이 아니라 제출 우선순위를 정하기 위한 조건부 시나리오다.

## P3 공식 무정답 후보

실험: `p3_champion_matched_era5_hs2_official_deploy_20260828_v1`

- source: ERA5 2014–2023 same-year complete footprint 10,492 cases, source fit 1회
- local: immutable 24,360 anchors, continuation fit 1회
- feature: frozen 286 past-only features
- seed: 20260824
- parameter search: 0
- output: 1,200 rows / 200 cases
- changed support: 18h·24h 400 rows 전부
- protected support: 3h·6h·9h·12h 800 rows champion numeric bit-exact
- finite·0–30 m·key/order·schema: PASS
- 공식 truth·절대시간 복원·외부 평가기간 관측 매칭·upload: 0
- 독립 QA 및 `validate_p3_submission.py`: PASS

후보 파일:

`C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P3_ERA5_HS2_CHAMPION_MATCHED_READY\P3_submission.csv`

SHA-256:

`3967333b790c06495dff619b2a8191b9bec18aa56dff1453ee31f77882ce8a50`

직렬화 주의사항: 최초 pandas CSV 출력에서 보호 대상 800행 중 4행이 최대 4.44e-16만큼 달라져 guard가 중단됐다. 모델 재학습·재추론·파라미터 변경 없이 champion 원문 CSV의 보호 대상 800행을 그대로 복사했고, 최종 후보는 numeric bit-exact다. 이 복구는 1회이며 결과·manifest에 기록됐다.

계약 정정 메모: 배포 config의 공식 입력 allowlist는 `test_context.parquet`와 `test_index.csv`를 명시하지만, 독립 형식 validator는 추가로 `sample_submission.csv`의 키 구조만 읽었다. sample의 예측값이나 hidden truth는 사용하지 않았고 통계도 최종 보고서에 기록하지 않았다. 따라서 성능 누출은 없지만, 엄밀한 파일 접근 기록상 sample schema read 1회는 존재한다.

## 최종 의사결정

1. P1: 제출하지 않는다. 현 champion 유지.
2. P2: 제출하지 않는다. OAS40 champion 유지.
3. P3: 이 파일만 공식 1회 probe 자격이 있다. 업로드 전 파일 경로와 SHA를 다시 대조한다.
4. 공식 결과 해석은 사전 고정한다.
   - +0.03점 이상: Public 개선으로 채택
   - -0.03~+0.03점: 무정보, 미세조정 금지
   - -0.03점 이하: 이 deployment family 종료
   - +0.10점 이상: 구조적 신호로 보존하되 Private-ready로 단정하지 않음

현재 업로드 횟수는 0이다.

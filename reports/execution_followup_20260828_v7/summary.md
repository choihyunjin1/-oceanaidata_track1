# 2026-08-28 후속 실행 결론 v7

## 결론

- **P1: 종료(NO-GO).** 기존 TS2Vec-style 인코더의 90.8087% coverage 실패는 512행 미만 segment를 버린 추론 구현 때문임을 확정했다. 별도 one-shot에서 472개 segment, 355,674행을 모두 추론해 coverage 100%를 달성했지만 calibration과 qualification 모두 TP=0, F1=0이었다. 따라서 coverage 문제를 제거해도 고정 prototype/long-component 구조에는 탐지력이 없었다.
- **P2: 공식 개선 확인 및 파일 무결성 유지.** OAS40 파일은 1,284,060 bytes, SHA-256 `6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96`로 기존 QA와 동일하다. 공식 결과는 RMSE 0.445147, 27.747847점이며 이전 최고 27.264587점 대비 **+0.483260점**이다. 이 축은 실제 공식 개선 증거를 남겼다.
- **P3: 다음 공식 검증 가치가 있는 후보 발견.** ERA5 transfer prediction의 18/24h `Hs²` residual만 incumbent에 25% 반영하고 나머지 724행은 bit-exact incumbent로 유지했다. 전체 RMSE는 0.7799487225에서 0.7763096877로 **0.0036390348m 개선**됐다. 3개 fold와 3개 station이 모두 개선됐고, complete-case bootstrap 90% CI는 `[-0.0061285, -0.0012620]m`, 개선확률은 0.995였다.

## P1 증거

- parent coverage: 0.90808718096
- recovery coverage: 1.0
- 512행 미만 segment: 363개, 32,691행
- calibration: TP 0, FP 0, FN 3,707, F1 0
- qualification: TP 0, FP 0, FN 4,080, F1 0
- result: `artifacts/p1_ts2vec_full_segment_coverage_recovery_20260828_v1/result.json`
- independent QA: PASS

판단: threshold나 component 길이를 결과에 맞춰 다시 조정하지 않는다. 이 representation-plus-prototype 계열은 종료한다.

## P2 증거

- candidate: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260828_P2_SEASONAL_OAS_TS40_PROJECTED_READY\P2_submission.csv`
- rows: 26,061
- official RMSE: 0.445147C
- official points: 27.747847
- score delta: +0.483260 points
- official receipt: `reports/p2_submit_p1_p3_deep_research_20260828_v1/official_score_receipt.json`

판단: OAS 강도축의 local exposed OOF 방향과 official 방향이 달랐으므로, P2는 local 점수만으로 공식 후보를 제거하면 안 된다는 보정 증거다.

## P3 증거

고정 공식:

`E_candidate = E_incumbent + 0.25 * (E_ERA5_transfer - E_incumbent)`, 단 `E = Hs²`, lead 18/24h만 적용.

| 구간 | incumbent RMSE | candidate RMSE | delta m |
|---|---:|---:|---:|
| 전체 | 0.7799487 | 0.7763097 | -0.0036390 |
| 2024 H2 storm | 0.7167315 | 0.7109796 | -0.0057519 |
| winter transition | 0.7867957 | 0.7847603 | -0.0020353 |
| 2025 H1 | 0.8245030 | 0.8202589 | -0.0042441 |
| lead 18h | 0.8929577 | 0.8852625 | -0.0076951 |
| lead 24h | 0.8474208 | 0.8353598 | -0.0120610 |
| G-ORS | 0.7280140 | 0.7213285 | -0.0066855 |
| I-ORS | 0.8870022 | 0.8837234 | -0.0032788 |
| S-ORS | 0.7515090 | 0.7504440 | -0.0010651 |

- 새 fit: 0회
- parameter search: 0회
- active rows: 362/1,086
- inactive rows: 724/1,086 bit-exact
- official/test rows read: 0
- result: `artifacts/p3_era5_longlead_energy_residual_shrink_20260828_v1/result.json`
- independent QA: PASS

한계: 이 구조는 parent outer 결과의 long-lead 신호를 본 뒤 제한한 adapted probe다. 따라서 완전히 독립적인 새 holdout 증거는 아니며, 다음 공식 1회가 실제 transport 검증 역할을 해야 한다.

## 다음 행동

1. P1 TS2Vec-prototype은 중단하고 다른 representation family로 이동한다.
2. P2 OAS40은 이미 공식 개선 증거가 있으므로 현 champion 기록으로 유지한다.
3. P3는 사용자에게 official candidate 생성에 필요한 정확한 test/sample 경로 읽기 및 CSV 생성 승인을 별도로 받은 뒤, 본 고정 공식을 변경하지 않고 한 번만 적용한다.
4. P3 공식 점수 전에는 weight, active lead, station 조건을 변경하지 않는다.

이번 실행에서는 공식 test/sample/submission 경로를 읽지 않았고, CSV 생성·업로드·commit·push도 하지 않았다.

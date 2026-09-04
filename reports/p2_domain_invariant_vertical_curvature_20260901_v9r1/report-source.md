# P2 domain-invariant vertical-curvature exploratory cycle 20260901 v9r1

## 결론

상태: `EXPLORATORY_NO_GO_BOTH_SEALED_CANDIDATES`. 세 historical fold는 모두 이미 노출된 exploratory surface이며 fresh confirmation이 아니다.

| 후보 | pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | raw 예상점수 | transport 보정점수 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| P2_V9_GLOBAL_ROBUST_NCR_RIDGE_BLEND020 | -0.012971020 | -0.036490247 | +0.052132376 | +0.036979929 | +0.330606 | +0.208924 | NO_GO |
| P2_V9_LAYER_MONTH_ANOMALY_NCR_RIDGE_BLEND020 | -0.006936923 | -0.038816677 | +0.099422554 | +0.036979929 | +0.351965 | +0.230283 | NO_GO |

## v7 기존 제출팩 보고서 감사

`P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE`는 보고서 원장상 `REPORT_ATTESTED_READY_UNSUBMITTED_HIGH_TRANSPORT_RISK`이다. 당시 daily limit로 upload attempted=false였고 exact family 후속 공식 제출 기록은 없다.
보정 기대치는 +0.013765점, 현 챔피언 기준 명목 RMSE 0.429097°C / 27.949229점이다.
CSV 값이나 bytes를 다시 읽지 않았고, 기존 independent root-ready QA의 schema/key/order/finite/hash attestation과 현재 파일 metadata 존재만 대조했다.

## 중복 배제와 안전장치

v8r1과 달리 raw-Celsius L1/L2 잔차를 학습하지 않는다. 공개층 기반의 무차원 normalized-curvature target을 layer별 Ridge로 학습하고, 결과 전 고정한 0.2 blend와 bounded delta로 챔피언 proxy를 0.8 보존한다.
두 번째 후보는 training-only layer-month median을 제거하며 unseen month에는 layer-global median만 사용한다. 행 삭제는 0이고 target에만 4-MAD fixed winsorization을 적용했다.
후보 postprocess, DINEOF/GP/CatBoost/soft-gate/PAVA/rank-season-bin search는 사용하지 않았다. comparator의 기존 projection lineage는 변경하지 않았다.

## 검증 경계

점수 환산은 기존 소규모 delta calibration의 명목값이며 공식 기대값이 아니다. 세 fold와 bootstrap은 후보 폐쇄와 우선순위 판단만 지원한다.
fits=6; official/test/sample/baseline/score/query/hidden rows=0; submission CSV=0; upload=0.

## 기술 contract repair

v9은 첫 model.fit 전 0-fit으로 종료됐다. v9r1은 presence indicator가 이미 0인 training-window all-missing layer-8 feature 네 열만 deterministic zero로 바꿨다. 후보, split, alpha, 0.8/0.2 blend, gate, 4-MAD winsor는 변경하지 않았다.

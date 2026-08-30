# 2026-08-31 모델별 과거 실험 전수 재감사

## 결론

**과거 결과를 다시 살펴본 결론은 “옛 탈락 후보를 전부 제출”도 “실패 모델 계열을 전부 폐기”도 아니다.** 저장된 48개 historical family를 최신 봉인 확인과 공식 정보까지 반영해 전수 재판정한 결과, 주상태는 exact recipe 종료 25개, 발견용 11개, 정보가치 양성 6개, 옛 gate 때문에 탈락 2개, 프록시 노출 3개, 기술 무효 1개다. exact 종료는 모델명 전체가 아니라 레시피·split·feature·postprocess·gate의 정확 조합만 닫는다.

과거 “재확인 가치 있음” 판정도 최신 근거로 덮어썼다.

- P1 Sobol trial18/threshold 0.8은 Q2의 최고점 후보였지만 sealed Q3/Q4에서 ΔF1 `-0.011889120`이므로 exact recipe는 닫았다. 다만 최고 체크포인트였다는 사실은 `CHECKPOINT_PEAK`, 선택면 수송 실패는 `PROXY_EXPOSED` 태그로 보존했다.
- P2 Gaussian copula v2는 historical proxy ΔRMSE `-0.010616065°C`가 official Public ΔRMSE `+0.012050°C`로 반전되어 exact recipe를 닫고 `PROXY_EXPOSED`로 기록했다.
- P3 CatBoost challenger21은 selection ΔRMSE `-0.0228625m`가 repaired confirmation에서 `+0.0079741m`로 반전되어 exact recipe를 닫았다.
- P3 lead-continuous는 fresh episode 1개에서 `+0.022617090m` 악화됐지만 독립 block이 1개뿐이므로 harm 확정 대신 `DISCOVERY_ONLY + PROXY_EXPOSED`로 남겼다.

## 전수 범위와 counting rule

이번 감사는 네 개의 겹치는 grain을 모두 대조했다.

| grain | 수 | 용도 |
|---|---:|---|
| historical family | 48 | 전수검사의 고유 분모; P1 17, P2 19, P3 12 |
| canonical group | 35 | exact closure와 invalid scope의 정규화 교차검사 |
| later key case | 20 | 2026-08-28~30 후속 실험과 최신 근거 반영 |
| workflow exception | 4 | 과학 결론이 없는 기술·식별 실패 분리 |

네 grain은 중복되므로 `48+35+20+4=107개 고유 실험`으로 합산하지 않는다. 48 family가 historical 전수 분모이고 나머지는 판정의 정합성을 확인하는 보조 grain이다.

## 새 상태체계

각 레코드는 배타적인 `primary_status` 하나와 중첩 가능한 `status_tags`를 갖는다.

| 상태 | 의미 |
|---|---|
| `CLOSED_EXACT` | 정확히 시험한 조합만 종료; broad model family는 미종료 |
| `INVALID_TECHNICAL` | dependency/schema/식별 실패로 과학 결론 없음 |
| `DISCOVERY_ONLY` | 기전·계보 연구에는 재사용 가능하나 독립 확인 또는 제출 준비 아님 |
| `OLD_GATE_REJECTED` | unsupported hard gate가 양수 또는 inconclusive 신호를 과잉 탈락시킴 |
| `CHECKPOINT_PEAK` | 중간 최고 checkpoint를 후보로 보존; final-epoch와 동일시 금지 |
| `INFORMATION_POSITIVE` | 공식 또는 의존성 보존 local 근거가 유의미한 방향 정보를 제공 |
| `PROXY_EXPOSED` | selection/proxy 신호가 sealed/official surface에서 반전 또는 약화 |

자세한 정의와 counting rule은 [status-taxonomy.md](./status-taxonomy.md), 전체 레코드와 fingerprint는 [candidate-ledger.json](./candidate-ledger.json)에 있다.

## 문제별 최신 판정

| 문제 | exact 종료 | 발견용 | 정보가치 양성 | 옛 gate 탈락 | 프록시 노출 | 기술 무효 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 9 | 2 | 3 | 2 | 1 | 0 |
| P2 | 13 | 4 | 1 | 0 | 1 | 0 |
| P3 | 3 | 5 | 2 | 0 | 1 | 1 |

이는 family의 **주상태** 집계다. 예를 들어 P1 trial18은 48-family 기본 원장과 별도의 later key case라 이 표의 P1 `PROXY_EXPOSED` 1개와 별도로 key-case 태그에 보존된다.

## 과거 탈락 중 아직 살아 있는 범위

무제한 재실행 대상은 없다. 아래는 exact frozen confirmation이나 새로운 외부 면이 있을 때만 재개할 수 있다.

- P1 block inpaint: pooled ΔF1 `+0.002591`, 기존 CI가 0을 가로질렀지만 unsupported worst-slice veto로 탈락했다.
- P1 dynamic peer reliability: micro ΔF1 `+0.004640`, 기존 CI가 0을 가로질렀고 weighted 효과는 작다. frozen confirmation 가설만 보존한다.
- P1 environment-balanced replay, segment-precision core, window-phase: 작은 양수 또는 package 내부 양수 신호였으나 옛 고정 floor/복합 veto가 막았다. 모두 low-priority frozen-only다.
- P2 supervised/cross-fit rank-1, nested PLS, state-conditioned copula: positive local mechanism evidence는 남지만 같은 proxy에서 추가 미세조정하지 않는다. 특히 copula 계열은 Gaussian v2 official reversal을 항상 함께 본다.
- P3 sparse-GP abstention과 lead-continuous: efficacy는 inconclusive다. coverage/lead 구조만 재사용하고 multiple fresh episode 없이는 승격하지 않는다.

## 공식 정보가 실제로 남긴 재사용 축

- P1: 고정 15개 G-ORS 추가행은 제거 대비 표시 Public F1 `+0.004519`; 238개 S-ORS 추가행의 표시 marginal은 `0`이었다. G라는 factor는 정보 양성이지만 rowwise truth나 Private 보장은 아니다.
- P2: rank-1 bin17-only가 prior champion 대비 Public RMSE `-0.000015°C`, 점수 `+0.000187`로 개선했다. 인접 bin18은 반대였으므로 season-bin을 묶지 않는다.
- P3: uniform KMA `alpha=0.425`가 prior champion 대비 Public RMSE `-0.000029m`, 점수 `+0.000473`을 얻었다. station/lead factor는 재사용하되 nearby-alpha sweep는 금지한다.

## 모델별 재사용 카드

[model-cards/README.md](./model-cards/README.md)에 14개 카드가 있다.

- P1 4개: event-router/anchor, temporal neural, imputation-boundary-peer, external-density transfer
- P2 5개: OAS/rank1/low-rank, deep/GBM/stack, physical/surrogate, copula, profile/analog/prequential
- P3 5개: persistence/KMA, CatBoost, analog/spectral/GP/lead, neural/RevIN/SSL, ERA5/context transfer

각 카드는 `재사용할 것`, `그대로 반복하지 않을 것`, `재개 조건`, `근거 레코드`를 분리한다. 다음 연구 프롬프트나 runner는 후보 생성 전에 card id와 fingerprint를 조회해 exact/semantic 중복을 막아야 한다.

## 재사용 프로토콜

1. 제안 후보를 문제와 모델 card에 연결한다.
2. 변형, split, feature, postprocess, gate, source path를 canonical JSON으로 만들고 SHA-256 fingerprint를 계산한다.
3. exact fingerprint가 `CLOSED_EXACT`면 실행하지 않는다.
4. semantic하게 같은 메커니즘이면 카드의 `do_not_replay`와 `reopen_trigger`를 대조한다.
5. `OLD_GATE_REJECTED`는 parameter sweep가 아니라 exact frozen confirmation만 허용한다.
6. `CHECKPOINT_PEAK`는 checkpoint를 보존하되 selection 성능을 confirmation으로 보고하지 않는다.
7. `INFORMATION_POSITIVE`는 performance lane과 분리하고 공식/Private 준비로 자동 승격하지 않는다.
8. `PROXY_EXPOSED`가 붙은 계열은 같은 proxy class에서 재튜닝하지 않고 evaluation surface를 먼저 바꾼다.
9. `INVALID_TECHNICAL`은 결함을 preflight한 새 preregistered attempt만 허용하며 기존 lock을 재사용하지 않는다.

## 독립 QA와 보존 경계

[independent-qa.json](./independent-qa.json)은 다음을 PASS했다.

- 48 family와 P1/P2/P3 `17/19/12` 일치
- 35 canonical group, 20 key case, 4 workflow exception 일치
- 모든 family/key case가 14개 model card 중 하나에 연결
- grain 내부 fingerprint 중복 0
- trial18, Gaussian copula, lead-continuous, ERA5의 최신 근거 override 확인
- 새 model fit 0, raw/prediction 행 읽기 0
- official test/sample/submission/hidden 값 읽기 0
- CSV 0, upload 0

따라서 이 문서는 과거 저장 결과의 재판정이지 새 성능 실험이 아니다. Public 표시값은 Private 성능 보장이 아니며 P1 F1, P2 °C RMSE, P3 m RMSE의 크기를 서로 직접 비교하지 않는다.

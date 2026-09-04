# P3 표적분포 전이 재감사 — 결론 및 다음 구조

## 결론

**P3의 다음 방향은 동일 Hs²·ERA5 보정축의 추가 미세조정이나 중요도 가중 승격이 아니다.** 현재 챔피언 OOF의 상승파 비율은 `82.87%`, 공식 무라벨 문맥은 `81.50%`로 이미 유사하다. 파고 상태만으로 두 집단을 구분한 교차검증 AUC도 `0.562`에 그쳤다. 반면 파고와 풍속을 함께 쓰면 AUC가 `0.726`으로 올라가며, 역사 OOF의 풍속 완비율은 `69.61%`뿐이다. 특히 I-ORS와 S-ORS의 2024년 대기자료가 비어 있는 반면 공식 문맥의 풍속은 거의 완비되어 있다.

더 중요한 반증은 **이미 공식에서 실패한 Hs² 후보가 세 가지 재가중 로컬 표면 모두에서 여전히 개선으로 판정됐다는 점**이다. 공식 결과는 챔피언 대비 RMSE `+0.001846m`, 점수 `-0.029302`였지만, 로컬 중요도 가중 델타는 전체 물리 특징 `-0.013263m`, 파고 상태 표면 `-0.016286m`, 풍속 완비 표면 `-0.011132m`였다. 따라서 관측 공변량만으로 계산한 중요도 가중치는 이번 P3에서 공식 전이를 검증하는 승격 게이트로 사용할 수 없다.

이 진단에서 정한 **풍속 완비 strict-OOF direct-residual pilot도 실제 실행했다.** Ridge 3종과 저심도 CatBoost가 모두 챔피언보다 악화했고 승격 게이트를 통과한 모델은 0개였다. 가장 덜 나쁜 풍속 전용 Ridge도 RMSE가 `+0.008173m` 악화했고 CI90은 `[-0.001025,+0.018175]m`였다. 3개 outer fold 중 2개는 내부 선택이 residual scale `0`을 골랐으며, `0.5`를 고른 2025 H1에서 악화했다. 따라서 **P3의 관측 풍속 잔차축도 종료**한다. 후보 CSV를 만들 이유가 없으며, 이 보고서에서도 공식 test label, sample/submission 경로, CSV 생성 및 업로드는 전혀 사용하지 않았다.

## 질문과 판정

| 질문 | 관측 | 판정 |
|---|---:|---|
| 상승파 선택 비율이 다른가? | OOF `150/181=82.87%`, test `163/200=81.50%` | 아니오. 기존 OOF가 이미 상승 국면 중심이다. |
| 파고 상태 분포가 크게 다른가? | domain AUC `0.562`, MMD² `0.01650`, p=`0.000999` | 차이는 검출되지만 분류력은 약하다. 주 병목으로 보기 어렵다. |
| 전체 관측 특징은 다른가? | domain AUC `0.726`, MMD² `0.06545`, p=`0.000999` | 예. 풍속 결측 구조가 가장 큰 차이다. |
| 역사 OOF에 풍속이 충분한가? | `126/181=69.61%`; G `66/67`, I `30/46`, S `30/68` | 불균형하다. 단순 결측 대치가 공식 문맥을 대변하지 못한다. |
| 중요도 가중이 공식 Hs² 실패를 예측했나? | 세 표면 모두 `ΔRMSE<0`, CI90 상한도 `<0` | 아니오. 공식 방향과 반대이므로 승격 기준으로 부적합하다. |
| 풍속 완비 잔차 모델이 개선했나? | 4개 모델 모두 `ΔRMSE>0`; 최선 `+0.008173m` | 아니오. 이 축을 종료한다. |

## 연구 근거

공변량 이동에서 중요도 가중 교차검증이 정당화되려면 입력 분포만 변하고 조건부분포 `P(Y|X)`가 안정적이어야 한다. Sugiyama 등은 이 가정 아래 IWCV를 제시했고, Bickel 등은 목표 분포가 학습 분포의 support 안에 있어야 한다는 전제와 판별적 보정을 다뤘다. 이번 공식 probe의 역전은 그 가정이 성립하지 않거나, 현재 특징이 전이에 필요한 상태를 담지 못하거나, 배포 변환이 역사 재현과 다르다는 실증적 경고다. [Sugiyama et al., JMLR 2007](https://jmlr.org/papers/v8/sugiyama07a.html), [Bickel et al., JMLR 2009](https://www.jmlr.org/beta/papers/v10/bickel09a.html)

MMD는 두 표본의 분포 차이를 커널 평균 임베딩으로 검사한다. 여기서는 차이의 존재를 확인하는 보조 진단으로만 사용했으며, 작은 p값을 곧바로 예측 성능 전이 보증으로 해석하지 않았다. [Gretton et al., JMLR 2012](https://www.jmlr.org/beta/papers/v13/gretton12a.html)

풍파 예측 문헌은 여러 예측 리드를 직접 출력하고 지역별로 서로 다른 과거 창을 쓰며, 풍속과 파고 사이의 지연을 명시적으로 다루는 구조를 보고한다. 이를 근거로 pilot에서는 캐시에 실제 존재하는 12시간 및 18/24시간 풍속 지연과 6개 리드별 챔피언 잔차를 연결했다. 다만 해당 논문의 성능을 이 데이터에 그대로 운반하지는 않는다. [Applied Sciences 2026 direct multi-horizon wave forecasting](https://www.mdpi.com/2076-3417/16/15/7447), [Ocean Engineering 2022 multi-step GRU](https://www.sciencedirect.com/science/article/abs/pii/S0029801822001469)

## 재감사 수치

### 분포 진단

| 진단 표면 | source 사례 | AUC | propensity ESS | NN ESS | MMD² / p |
|---|---:|---:|---:|---:|---:|
| 파고+풍속 전체 | 181 | 0.7261 | 97.17 | 61.54 | 0.06545 / 0.000999 |
| 파고 상태만 | 181 | 0.5624 | 97.52 | 74.91 | 0.01650 / 0.000999 |
| 풍속 완비 source | 126 | 0.6262 | 66.22 | 59.00 | 0.01778 / 0.003996 |

### 공식 반증이 있는 후보 재평가

`ΔRMSE<0`은 로컬 개선을 뜻한다.

| 평가 표면 | Hs² 가중 ΔRMSE | 90% bootstrap CI | 공식 실제 방향 |
|---|---:|---:|---|
| 전체 물리 특징 | -0.013263m | [-0.01884, -0.00786] | 악화 |
| 파고 상태만 | -0.016286m | [-0.02402, -0.00982] | 악화 |
| 풍속 완비 source | -0.011132m | [-0.01675, -0.00581] | 악화 |

ERA5 long-lead residual shrink와 advantage router는 풍속 완비 표면에서 각각 `-0.000739m`와 `+0.008873m`였고, 두 CI90 모두 0을 가로질렀다. TimeXer, Chronos-2, event NLinear는 재가중 후에도 챔피언보다 명확히 나빴다. 따라서 기존 후보군에서 새로 승격할 모델은 없다.

## 실행 결과: atmosphere-complete residual pilot

목표는 **모델을 크게 만드는 것**이 아니라 **공식 문맥과 동일하게 풍속이 관측된 조건에서, 현재 챔피언의 남은 오차가 풍속 지연으로 재현 가능하게 줄어드는지** 검증하는 것이었다.

1. 현재 챔피언 strict OOF 181사례 중 현재·12/18/24h 풍속 및 요약 특징이 모두 존재하는 124사례만 사용했다.
2. 목표는 6개 리드별 `actual_hs - champion_prediction`으로 두고 챔피언 자체는 고정했다.
3. 과거 파고, 관측 풍속·돌풍의 12h 및 18/24h 지연, station indicator만 사용했다. 공식 test feature도 읽지 않았다.
4. Ridge wave-only, wind-only, wave+wind와 depth 2~4 CatBoost wave+wind를 비교했다.
5. alpha/depth/L2와 residual scale은 각 outer training fold 안의 나머지 historical fold만으로 선택했다.
6. 전체 델타, case-bootstrap CI90, station/fold 일관성, 18/24h를 사전 등록 게이트로 검사했다.

| 모델 | ΔRMSE | CI90 | 18/24h ΔRMSE | 판정 |
|---|---:|---:|---:|---|
| Ridge wind-only | +0.008173m | [-0.001025, +0.018175] | +0.009698m | FAIL |
| Ridge wave+wind | +0.009115m | [-0.000363, +0.019541] | +0.010918m | FAIL |
| Ridge wave-only | +0.013649m | [+0.002888, +0.025527] | +0.018666m | FAIL |
| CatBoost wave+wind | +0.020167m | [+0.006734, +0.034951] | +0.030259m | FAIL |

풍속 전용 Ridge는 G/I/S에서 각각 `+0.003890`, `+0.004217`, `+0.023262m` 악화했다. 내부 선택은 2024 H2 storm과 winter transition의 residual scale을 `0`으로, 2025 H1만 `0.5`로 골랐다. 신호가 특정 구간에서만 보인 것이 아니라, 실제 보정을 허용한 구간에서 일반화가 실패한 것이다.

## 다음 자원 배분

- P3에서는 Hs²/ERA5 재가중축과 관측 풍속 잔차축을 모두 닫는다.
- 같은 입력 표현의 추가 파라미터 최적화나 epoch 증가는 실행하지 않는다. 신호 부재를 계산량으로 해결할 근거가 없다.
- P3는 방향·주파수 스펙트럼, 해류/수심, 미래 forcing처럼 **현재 1277개 과거 요약 특징과 독립적인 정보**가 확보될 때만 다시 연다.
- 다음 딥리서치·실행 자원은 P1/P2의 아직 닫히지 않은 구조적 축에 우선 배분한다.

## 제한과 중단 기준

- 공식 test label은 접근할 수 없으므로 `P(Y|X)` 안정성은 직접 검증할 수 없다.
- 현재 풍속·돌풍만 완비된 source는 126사례이고, 확장 지연 특징까지 완비된 pilot 표본은 124사례다. station별 표본도 불균형해 복잡한 모델의 분산이 커질 수 있다.
- MMD p값은 분포 차이의 존재를 말할 뿐, 어떤 모델이 공식 점수를 올릴지는 말하지 않는다.
- 공식 Hs² probe 한 번이 모든 물리 보정의 실패를 증명하지는 않지만, 동일 축 미세조정과 재가중 승격을 중단하기에는 충분한 반증이다.
- residual pilot에서 안정적 신호가 없었으므로 P3 풍속축을 닫고 P1/P2의 더 큰 기대가치 구조로 자원을 이동한다.

## 산출물

- 실행 설정: `configs/experiments/p3_target_shift_retroaudit_20260828_v1.json`, `v2.json`
- 실행 코드: `scripts/run_p3_target_shift_retroaudit_20260828_v1.py`, `v2.py`
- 결과: `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json`, `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json`
- 잔차 pilot 설정·코드·결과: `configs/experiments/p3_atmosphere_complete_residual_pilot_20260828_v1.json`, `scripts/run_p3_atmosphere_complete_residual_pilot_20260828_v1.py`, `artifacts/p3_atmosphere_complete_residual_pilot_20260828_v1/result.json`
- 검증: `independent_qa.json`
- 주장-근거 원장: `claim-source-ledger.md`
- 공백표: `gap-matrix.md`

# P2 공식 검증 및 P1·P3 구조 연구 결론

작성일: 2026-08-28 (KST)
범위: P2 공식 1회 검증, P1/P3 신규 로컬 실험, 구조적 병목 분석, 다음 승격 후보 결정

## 결론

1. **P2 제출은 성공했다.** alpha40 후보는 Public RMSE `0.445147°C`, 공식 점수 `27.747847/33`을 기록했다. 직전 최고 `0.483661°C`, `27.264587점` 대비 RMSE는 `0.038514°C`(약 `7.96%`) 감소했고 점수는 `+0.483260` 상승했다. 채점 직후 P2 잔여 기회는 `2/3`이다.
2. **P1의 “자원 부족”은 컴퓨터 자원 부족이 아니다.** frozen generator가 만든 학습 가능한 utility-positive event가 train `2/10`, calibration `0/4`로 부족하고 train 양성 사건이 한 station×layer cell에 `100%` 몰린 통계 표본 부족이다.
3. **P1의 NCAD-inspired synthetic TCN은 아이디어를 학습했지만 실관측으로 운송되지 않았다.** selection에서는 F1이 `+0.007025`였으나 calibration에서 `-0.371438`, qualification에서 `-0.117604`로 악화하고 추가 TP가 모두 0이었다.
4. **P3의 past-only 구조는 장기 lead에서 반복적으로 막혔다.** compact NLinear/DLinear-inspired ridge는 후보 RMSE `0.785851m`로 incumbent `0.779949m`보다 `0.005902m` 악화했고, 12/18/24h에서 모두 악화했다. 기존 TSMixer도 같은 방향이었다.
5. **P3 ERA5 context-transfer는 모델 실패가 아니라 미실행 상태다.** raw `363/363`, derived `363`, combined `262,917행`, 286-feature preflight까지 통과했지만 download-only interpreter에 CatBoost가 없어 fit `0회`로 종료됐다. 따라서 이 계열은 폐기할 과학적 근거가 아직 없다.

## P2 공식 검증

### 제출 파일

- 제목: P2 계절 국소 T/S 조건부 프로파일 OAS 40% v1
- 접근: 기존 U 60% + 계절 국소 OAS T/S 조건부 프로파일 40% + endpoint/PAVA 물리 투영
- 행 수: 26,061
- 열: `station, layer, time, temp`
- SHA-256: `6e28ddb8d78c0969e5104d7efbe28e1762f51e80d759fceb86cdef52baa29b96`
- QA: 키 유일성·순서, finite/range, layer별 행 수, PAVA idempotence 모두 PASS

### 공식 결과

| 항목 | 직전 최고 | alpha40 | 변화 |
|---|---:|---:|---:|
| Public RMSE (°C) | 0.483661 | 0.445147 | -0.038514 |
| 공식 점수 | 27.264587 | 27.747847 | +0.483260 |

사전 추정 중심 `0.448627°C`와 실제 `0.445147°C`의 차이는 `-0.003480°C`로, same-lineage 예측 기하가 이 축에서는 유용했다. 반면 exposed local OOF는 alpha40이 alpha20보다 `0.016402°C` 악화한다고 예측했다. 즉 P2에서 로컬 surrogate의 절대 순위는 공식 surface와 어긋났고, 동일 계보의 공식 벡터 기하가 더 잘 운송됐다.

### 운영 결정

- alpha40을 새 P2 Public incumbent로 보존한다.
- 남은 2회는 alpha60/80의 자동 연장에 쓰지 않는다.
- 다음 P2 제출은 fresh temporal holdout 또는 명확히 다른 구조가 alpha40을 이길 때만 승인한다.

## P1: 통계적 support 병목

### 정확한 의미

19행 이상 connected event proposal은 총 123개였으나, 학습 구간 45개 중 utility-positive는 2개뿐이었다. calibration은 proposal 2개 중 utility-positive 0개였다. hard-negative 비율 21.5는 기준 5.0을 통과했지만, 양성 사건이 한 station×layer cell에 100% 집중되어 허용치 70%를 넘었다. 그래서 verifier는 fit 0회로 support preflight에서 멈췄다.

이 실패는 GPU·RAM·시간·토큰 부족이 아니라 **정답을 가르칠 수 있는 다양하고 분산된 양성 사건 부족**이다. 모델을 크게 만들거나 epoch를 늘려도 표본 수와 분포 문제는 해결되지 않는다.

### NCAD-inspired 검증

NCAD 문헌의 contextual outlier exposure 아이디어를 48시간 causal TCN, synthetic offset/drift/noise/flatline/COE로 구현해 한 번 검증했다. selection에서는 후보가 anchor F1 `0.752104`에서 `0.759129`로 상승했지만, calibration에서는 284개를 추가해 TP 0 / FP 284, qualification에서는 TP 0 / FP 337이었다. 판정은 `NO_GO_CALIBRATION_SAFETY`다.

NCAD가 제안하는 합성 contextual anomaly 학습 자체는 문헌상 타당하지만, 현재 generator는 station×layer별 실제 이상 형태와 충분히 일치하지 않았다. 따라서 다음 P1 연구의 선행조건은 더 큰 네트워크가 아니라, 물리적으로 타당한 station×layer-conditioned event generator 또는 prospectively labeled positive event 확보다.

## P3: past-only 장기 lead 한계와 ERA5 미완료

### 신규 compact linear 검증

336개 past-only multi-resolution feature와 station별 standardized multi-output ridge를 사용했다. alpha는 inner validation에서 1/10/100/1000을 비교했고 세 fold 모두 1000을 선택했다. 3/6/9h는 incumbent를 보호하고 12/18/24h에 20% residual blend를 적용했다.

| 구간 | incumbent RMSE | candidate RMSE | 변화 |
|---|---:|---:|---:|
| pooled | 0.779949 | 0.785851 | +0.005902 |
| 12h | 0.864363 | 0.872553 | +0.008190 |
| 18h | 0.892958 | 0.904090 | +0.011132 |
| 24h | 0.847421 | 0.859850 | +0.012429 |

bootstrap 개선확률은 `3.62%`, 90% CI는 `[+0.000438, +0.011481]m`였다. 세 station 모두 악화했다. prior TSMixer도 같은 장기 lead에서 악화했으므로, 단순히 모델 용량이 작아서라기보다 past-only 입력에서 미래 forcing 정보가 빠진 병목이라는 해석이 더 강하다.

### ERA5 계열의 현재 상태

- raw 완료: 363/363, partial 0
- derived 완료: 363
- combined: 262,917행, SHA-256 `5106c4ee35c7d434dcea13d1b436691eea9b05ef9f8c59fdd900d4c19bad9ac1`
- preflight: PASS, common feature 286, source quarantine ready
- model fit: 0
- 실패: `ModuleNotFoundError: No module named 'catboost'`

이 결과는 ERA5 context-transfer가 나쁘다는 결과가 아니다. 모델을 한 번도 fit하지 못했으므로, 다음 P3 최고가치 실험은 **새 experiment ID**로 기존 frozen 286-feature/source-local split/gate를 그대로 유지하고, attempt lock 전에 exact interpreter의 CatBoost/sklearn/numpy/pandas/pyarrow import와 버전을 검증한 뒤 실행하는 것이다.

## 문헌이 주는 구조적 판단

- NCAD는 contextual outlier exposure로 unsupervised anomaly problem을 supervised form으로 바꾸는 방향을 제시한다. 이번 결과는 아이디어 자체보다 local generator-to-reality transport가 실패했음을 보여준다.
- DLinear/NLinear은 강한 단순 baseline이지만 이번 P3에서 장기 lead를 개선하지 못했다.
- TiDE는 covariate를 포함하는 MLP encoder-decoder이고, TimeXer는 exogenous information을 명시적으로 통합한다. P3 local 결과와 합치면 다음 돌파구는 더 큰 past-only backbone보다 future/exogenous forcing의 정확한 투입이다.

## 다음 실행 우선순위

1. **P3 fresh ERA5 frozen-contract attempt:** 가장 높은 우선순위. 환경 import/version preflight를 attempt lock 앞에 둔다. 모델·286 features·split·postprocess·gates는 바꾸지 않는다.
2. **P1 support generation redesign:** station×layer-conditioned physical injection 또는 prospective positive event ledger를 먼저 만든다. support gate 전에는 classifier capacity search를 하지 않는다.
3. **P2 incumbent 보존:** alpha40을 기준점으로 고정하고, 남은 공식 2회는 새 구조가 생길 때까지 보존한다.

## 재현성과 제한

- P1/P3 신규 실험은 공식 test/sample/submission을 읽거나 생성하지 않았다.
- P1 qualification truth는 support gate가 실패한 verifier 실험에서 열지 않았다.
- P2 Public 점수는 인증된 공식 페이지의 채점 카드 관측값이며 Private 점수는 대회 종료 후 공개된다.
- P1/P3의 로컬 결론은 고정 historical surface에 대한 결론이며 전체 모델 공간의 최대치를 증명하지 않는다.

## 외부 근거

1. IJCAI 2022 NCAD: https://www.ijcai.org/proceedings/2022/394
2. DLinear official implementation: https://github.com/honeywell21/DLinear
3. TiDE: https://openreview.net/pdf?id=pCbC3aQB5W
4. TimeXer, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf

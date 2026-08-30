# 원본 training 데이터 기반 돌파구 연구 결론

## 결론

**이번 사이클에서 새 제출 후보는 나오지 않았다.** 세 문제 모두 공식 test·sample·baseline·score·submission·hidden answer를 읽지 않은 training-only 연구였고, CSV 생성과 upload도 모두 0이었다. 따라서 현재 제출본을 교체하거나 추가 공식 평가를 요청할 근거가 없다.

| 문제 | Stage-0 | Stage-1 | 최종 판정 |
|---|---|---|---|
| P1 | `NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT` | 실행하지 않음 | Q4에서 utility-positive event가 `3 < 4`였으므로 residual add-only lane을 0 fit에서 종료 |
| P2 | `PASS`: 6/6 state cell, 7 dependence edge | `NO_GO_STATE_CONDITIONED_COPULA_STAGE1` | pooled·3개 layer·bootstrap은 개선했지만, 1/3 window만 엄격 개선했고 JJA worst-season guard를 `0.000174240°C` 초과한 near-miss |
| P3 | `PASS`: canonical selection-matched support 충족 | `NO_GO_CLOSE_THIS_EXACT_RECIPE` | candidate가 paired incumbent보다 `+0.314155m` 악화했고 3/3 window와 6/6 lead가 모두 악화한 decisive NO_GO |

Stage-0 PASS는 “구조를 시험할 support가 있다”는 뜻일 뿐 성능 승격이 아니다. P2와 P3는 그 support를 실제 frozen historical comparison으로 연결했지만 둘 다 사전등록 gate를 통과하지 못했다. P1은 support gate에서 멈췄으므로 model fit 자체를 허용하지 않았다.

## 문제별 결과

### P1 — heterogeneous event utility

Stage-0는 이미 동결된 다섯 OOF proposal source를 사건 단위로 합치고, incumbent의 음성 영역에 event를 add-only로 더할 실질 효용이 반복되는지만 확인했다. model fit, threshold search, prediction materialization은 모두 0이었다.

- `q2_to_q3` prefix는 통과했다. fit/calibration utility-positive event는 각각 `18/40`이었고 provenance도 완전했다.
- 그러나 `q2_q3_to_q4`에서 Q4 calibration utility-positive event는 최소 `4`개에 못 미치는 `3`개였다.
- Q4 proposal precision은 `0.102459`로, incumbent F1에서 유도한 필요 precision `F1/2 = 0.457171`보다 크게 낮았다.
- anchor positive row와 target-positive event를 제거하지 않았다. `anchor_positive_removed_rows=0`, model fit `0`, official interface row `0`, CSV/upload `0/0`이다.

따라서 현재 동결 proposal bank를 작은 residual head로 학습시키는 exact lane은 닫는다. Q2/Q3의 양성 신호만 보고 Q4 실패 뒤 threshold나 source 조합을 다시 고르는 것은 이미 노출된 validation에 대한 결과 기반 선택이 된다.

#### P1 실행 후 hardening chronology

P1 receipt가 생성된 뒤 runner에는 세 가지 방어적 변경이 추가됐다: exclusive receipt creation, concentration 경계의 exact rational 비교, provenance boolean의 엄격 검증이다. 현재 runner SHA256은 `cba95fb7577c04485063a643f8520978cc998ccc1b2fb100c06856b3e5537cb2`지만, 최초 실행 runner의 byte SHA256은 receipt에 남아 있지 않다. 이 hardening 뒤 같은 실험을 다시 실행하지 않았다.

따라서 최초 receipt가 “현재 runner와 byte-for-byte 동일한 코드로 실행됐다”고 주장하지 않는다. 대신 독립 QA가 이미 기록된 aggregate 값으로 gate를 재계산했다. Q4의 `3 < 4`, proposal precision `0.102459 < 0.457171`, 최대 cell share 등은 변경된 경계에서 충분히 떨어져 있어 NO_GO는 불변이다. 독립 QA 상태는 `PASS_WITH_DISCLOSED_P1_POST_RUN_HARDENING`이다.

### P2 — state-conditioned Gaussian-rank dependence

#### Stage-0 support PASS

공개 endpoint T/S로 정의한 thermal tertile × 24시간 변화 상태의 6개 cell이 모두 `≥500 profiles`, `≥30 KST days`, `≥2 chronological blocks`를 충족했다. 동일 state-cell pair의 Kendall τ 차이가 같은 방향으로 두 블록 이상 반복되는 edge는 사전 최소 2개보다 많은 7개였다.

고정 exposure edge는 다음과 같다.

1. `temp_contrast_signed__residual_l2`
2. `temp_contrast_signed__residual_l3`
3. `temp_contrast_signed__residual_l4`
4. `psal_contrast_signed__residual_l4`
5. `thermal_change_24h_signed__residual_l4`
6. `residual_l2__residual_l4`
7. `residual_l3__residual_l4`

Stage-0는 zero-fit support audit이다. 이 결과로 Gaussian-copula family의 예측력이 입증된 것은 아니다.

#### Stage-1 near-miss NO_GO

Stage-1은 이전에 닫힌 seasonal empirical-margin + shrinkage grid + Gauss–Hermite exact recipe를 재실행하지 않았다. 대신 outer-train에서만 만든 6개 public-state cell, 위 7개 sparse dependence edge, 고정 diagonal shrinkage `0.80`, unsupported/OOD exact no-op 한 가지 recipe를 세 frozen historical window에 한 번 적용했다.

| 구분 | Reference RMSE | Candidate RMSE | ΔRMSE |
|---|---:|---:|---:|
| pooled, 69,850 rows | 3.085786848 | 3.082327672 | **-0.003459176** |
| 2024 Sep–Oct | 4.882261511 | 4.875663356 | **-0.006598154** |
| 2025 Jul–Aug | 1.185757449 | 1.188931689 | **+0.003174240** |
| 2025 Nov–Dec | 0.280732185 | 0.280732185 | **0.000000000** |

Layer 2/3/4의 ΔRMSE는 각각 `-0.002769576`, `-0.005731568`, `-0.004014841°C`로 모두 개선했다. KST-day paired bootstrap CI90도 `[-0.006529882, -0.001922933]°C`, 개선확률 `1.0`이었다.

그럼에도 사전등록 gate 두 개가 실패했다.

- 엄격 개선 window는 2024 Sep–Oct 하나뿐이다. 2025 Nov–Dec의 exact tie는 개선으로 세지 않으므로 최소 `2/3`을 충족하지 못했다.
- JJA ΔRMSE `+0.003174240°C`가 worst-season cap `+0.003°C`를 `0.000174240°C` 초과했다.

2025 Nov–Dec에는 preregistered exact-24h public state가 유한한 profile이 없어 5,631개 profile 전부가 exact no-op이 됐다. 이는 이전 seasonal copula의 Nov–Dec `+0.034267°C` 악화를 반복하지 않은 안전장치지만, 해당 계절에서 새로운 개선 근거도 만들지 못했다. 결과를 본 뒤 JJA gate, 24시간 lag, shrinkage 또는 OOD quantile을 바꾸지 않는다.

Stage-1 fit은 outer model `3`, inner selection `0`, cell-correlation estimation `18`이었고 runtime은 `23.424s`였다. correction RMS/p99는 `0.046050/0.200000°C`로 cap을 통과했다. 결론은 pooled signal이 남아 있어 Gaussian conditional-dependence 가설 전체를 닫는 것이 아니라, **이번 discrete state-cell + fixed sparse correlation exact recipe를 승격하지 않는다**는 것이다.

Receipt에는 실행 1회, 결과 기반 tuning/retry 0이 기록돼 있다. 다만 P2 runner는
fit 전에 attempt lock을 소비하지 않고 모든 fit 뒤 aggregate result만 exclusive-create한다.
따라서 이는 **관측된 one-shot**이지 crash 이전 재실행까지 코드가 fail-closed로
막았다는 뜻은 아니다. 또한 Stage-1 runner hash는 receipt에 암호학적으로 결합돼
있지 않다. 이 provenance 한계는 성능 NO_GO를 바꾸지는 않지만 다음 runner에서는
fit 전 lock으로 고쳐야 한다.

### P3 — selection-matched masked SSL

#### Stage-0 support PASS와 243/242 정정

canonical cohort는 공식 lead `3/6/9/12/18/24h`, 48시간 elapsed history, `1.5≤Hs<2.2`, 12시간 상승 `>0.2m`, station-global 78시간 간격을 사용한다. 이 계약에서 global independent count는 **243**이며 station별로 `G/I/S = 86/68/89`다. applicable historical window support는 `41/51/65`로 모두 최소 20을 넘었다.

별도의 strict-complete 진단은 `t-48h` 시점의 Hs가 정확히 finite일 것을 추가 요구한다. 이때 count는 **242**, `G/I/S = 86/68/88`이다. 따라서 242는 canonical 243을 대체하는 값이 아니라, masked missing history를 허용하지 않는 더 엄격한 민감도 진단값이다. 같은 이유로 dense selection count도 canonical `2,131`, strict-complete `2,117`로 구분한다. 과거 초안에서 242를 canonical처럼 쓴 표현은 이 구분으로 정정한다.

#### Stage-1 decisive NO_GO

Stage-1은 157개의 동일 selection-matched anchor와 6개 공식 lead에서 masked SSL + frozen Huber head를 paired incumbent와 비교했다.

- overall candidate RMSE `1.125374m`, paired incumbent `0.811219m`, Δ `+0.314155m`
- paired-case bootstrap CI90 `[+0.244849, +0.388786]m`
- 개선 window `0/3`; window별 Δ는 `+0.380669`, `+0.089558`, `+0.429714m`
- 개선 lead `0/6`; 모든 lead가 악화했고 worst lead Δ는 `+0.383333m`

실제 fit budget은 masked encoder `3`, Huber head `3`, fixed reference router `2`, CatBoost fit `0`, 총 `8`이었고 runtime은 `37.788s`였다. 각 fold의 Huber가 500 iteration에서 convergence warning을 하나씩 남겨 integrity check 하나도 실패했다. 그러나 이 경고를 수리하면 양성이라고 주장할 수 있는 경계 사례가 아니다. 세 window와 여섯 lead의 부호가 모두 악화이고 bootstrap interval 전체가 0보다 크므로 exact recipe는 성능 기준만으로도 명확한 NO_GO다. 더 많은 iteration으로 같은 one-shot recipe를 재실행하지 않는다.

## 이상점과 극값 처리

이번 사이클의 공통 원칙은 **flag·weight는 진단에만 쓰고 hard deletion은 하지 않는 것**이었다.

- P1: `label=1` event와 anchor positive row를 제거하지 않았고 raw anomaly signal을 clip하지 않았다.
- P2 Stage-0: sensor-suspect profile 5,885개는 diagnostic-only였고, physical-extreme profile 1,752개를 모두 보존했다. Stage-1에서도 diagnostic weight는 state threshold, support count, Kendall fit, metric에 들어가지 않았고 hard deletion은 0이었다.
- P3 Stage-0: jump-and-return Hs flag는 2개였지만 cohort membership이나 weighting에 쓰지 않았다. `Hs≥2.2` storm-extreme anchor 3,507개를 보존했고 source row 삭제·mask는 0이었다. Stage-1도 high-wave/rapid-rise row 삭제가 0이었다.

이는 모든 flagged row가 정상이라는 뜻도, 모든 sensor error를 찾아냈다는 뜻도 아니다. QARTOD가 range, spike, rate-of-change, flat-line, multivariate test를 구분하듯, 극값 자체와 단일 센서 오류를 분리하기 위한 보수적 분석 계약이다. 특히 P1 anomaly, P2 thermocline/water-mass boundary, P3 storm extreme는 예측해야 할 신호일 수 있어 결과를 본 뒤 삭제 대상으로 바꾸지 않았다.

## 접근·산출물 감사

| 항목 | 결과 |
|---|---:|
| official test/test_index/sample/baseline/score/submission/hidden-answer rows read | 0 |
| query/context support rows read | 0 |
| prediction/submission CSV created | 0 |
| submission generated | 0 |
| upload attempted | 0 |
| source training rows deleted or modified | 0 |

P1은 이미 봉인된 OOF aggregate를 읽은 zero-fit preflight였다. P2는 `README.md`와 `observations.csv`만, P3는 `README.md`, `train_wave.csv`, `train_atmos.csv`만 명시 allowlist로 열었다. Stage-1 결과는 aggregate JSON만 남겼고 raw prediction row를 제출 파일로 만들지 않았다.

따라서 **현재 공식 제출 후보는 없다.** P2 near-miss를 기존 제출본에 섞거나 P3 candidate를 제출 surface로 변환하지 않는다.

## 한계

1. 모든 Stage-1 historical window, P2 edge, P3 cohort는 이미 연구에 노출됐다. PASS였더라도 공식 일반화 증거가 아니라 research-only evidence였을 것이다.
2. P2 comparator는 frozen alpha50 historical proxy이며 공식 incumbent의 exact historical OOF가 아니다. P3 paired 157-case surface도 기존 champion headline의 181-case RMSE `0.779105m`와 직접 비교할 수 없다.
3. P1 Stage-0는 이미 존재하는 proposal bank의 support를 검사했을 뿐 새로운 model 성능을 측정하지 않았다. 또한 post-run hardening 전 실행 runner의 byte hash가 없다는 provenance 한계가 있다.
4. P2 exact-24h state는 2025 Nov–Dec를 전부 no-op으로 만들었다. 안전성은 높였지만 그 regime에서 conditional dependence를 검증하지 못했다.
5. P3 Stage-1 Huber convergence warning은 integrity 실패다. 다만 performance degradation이 매우 커 이를 단순 기술 실패로 돌려 같은 recipe를 재시도할 근거는 없다.
6. sensor-suspect flag는 진단 규칙이지 오류 정답 label이 아니다. 삭제하지 않은 분석도, 향후 고정-rule sensitivity arm도 인과적 QC 판정을 대신하지 않는다.
7. 유한하고 반복 노출된 local validation에서 후보를 계속 고르면 selection bias가 커진다. 여기서 보고된 작은 P2 양성 신호를 새 hyperparameter search의 목적함수로 사용하지 않는다.
8. 커밋되는 independent-QA JSON은 로컬 immutable receipt의 hash와 재계산 결과를
   보존한 snapshot이다. `/artifacts/`가 ignore되므로 clean clone만으로 QA script를
   다시 실행할 수 없으며 원 receipt를 대체하는 완전한 재현 패키지는 아니다.

## 다음 가설과 우선순위

### 1. 새 holdout 또는 새 시점 확보

가장 중요한 다음 자원은 더 많은 architecture가 아니라 **현재 선택에 노출되지 않은 평가 시점**이다. 새 holdout 없이 P1 proposal 조합, P2 state 경계, P3 encoder/head를 다시 고르면 local selection bias만 누적된다.

### 2. P2 — availability-aware continuous state preflight

P2에서 pooled·layer·bootstrap 신호와 SON 개선은 남았다. 다음 가설은 discrete 3×2 cell을 더 잘게 튜닝하는 것이 아니라, public state에 따라 edge effect가 부드럽게 변하고 그 부호가 training block에서 사전 예측 가능하다는 것이다.

- 먼저 6h/12h/24h public dynamic coordinate의 availability와 동일-edge 부호 반복성을 0-fit으로 비교한다.
- shorter-lag를 선택하려면 Stage-1 결과가 아니라 사전 고정 support·direction gate를 사용한다.
- 연속 varying-coefficient 또는 다른 distributional learner는 새 holdout에서만 평가하고, support/OOD에서는 exact no-op을 유지한다.

현재 discrete-cell sparse Gaussian-rank exact recipe와 이전 seasonal empirical-margin recipe는 재실행하지 않는다. Gaussian-copula family 전체는 닫지 않는다.

### 3. P1 — proposal reliability shift를 먼저 반증

Q4 precision 붕괴가 station-layer, season, event duration 중 사전에 관측 가능한 어떤 축과 반복적으로 연결되는지 0-fit reliability audit부터 한다. 최소 precision lower bound가 여러 prefix에서 유지되지 않으면 새 residual head를 fit하지 않는다. 이미 노출된 Q4를 이용한 threshold 재선택은 금지하고, 가능하면 새로운 시간 block을 확보한다.

### 4. P3 — 새 정보가 없는 representation 확장을 중단

masked SSL exact recipe는 닫는다. Huber iteration만 늘리거나 encoder width를 키우는 재실행도 하지 않는다. 다음 후보는 다음 두 조건 중 하나를 먼저 충족해야 한다.

- past-only residual이 station/lead별로 반복 가능한 autocorrelation 또는 regime signal을 갖는다는 0-fit 증거
- 합법적이고 시점 정합적인 미래 forcing가 incumbent에 없는 정보를 실제로 추가한다는 별도 data-contract 및 held-out 증거

둘 다 없으면 paired incumbent의 exact no-op을 유지하는 것이 합리적이다.

## 근거와 해석 원칙

로컬 판정의 직접 근거는 다음 aggregate artifact다.

- `artifacts/p1_heterogeneous_event_utility_preflight_20260830_v1/terminal_result.json` — SHA256 `76a513f6b78a1ea94b6965c532524fc985feb25d4d51b02c147c5544e836d39a`
- `artifacts/p2_state_conditioned_copula_preflight_20260830_v1/result.json` — SHA256 `62fb0d97028fec1c299c333110625563149a76bb5187bbc623be6903af95cebd`
- `artifacts/p2_state_conditioned_copula_20260830_v1/result.json` — SHA256 `493ed58d5e7893f7c9297f26fe992393fdd3498820b562e1f045a77e1e992e2f`
- `artifacts/p3_selection_matched_cohort_preflight_20260830_v1/preflight.json` — SHA256 `4552364ddbd21e0674c6e69afc74516d3845126de84ae8f8cc06b4c0a25b0914`
- `artifacts/p3_selection_matched_masked_ssl_20260830_v1/result.json` — SHA256 `b6373cf4a6b281096dda4dbf10caf8bc9bebd01b01883269e421379ae702f7f5`
- `reports/original_dataset_breakthrough_research_20260830_v1/independent-qa.json`
- `reports/original_dataset_breakthrough_research_20260830_v1/stage1-independent-qa.json`
- `reports/original_dataset_breakthrough_research_20260830_v1/claim-source-ledger.md`

`artifacts/` 아래 실행 receipt와 attempt lock은 저장소 정책상 커밋하지 않는다. 대신
두 independent-QA JSON에 판정 재계산, fit/runtime, 원 receipt SHA256, 현재
config·runner·test·module SHA256, 고정 read surface를 보존했다. P3는 attempt
lock 내용과 result의 embedded one-shot attempt, 실행 implementation hash까지
일치했다. clean clone에서는 이 snapshot과 코드를 읽을 수 있지만 ignored 원
receipt가 없으므로 QA script 자체의 재실행이나 receipt 동일성 대조는 별도 로컬
artifact 보존본이 있어야 가능하다.

문헌은 후보를 생각하게 한 근거이지 로컬 성능 승격 근거가 아니다. P1 F1 precision floor 해석은 Lipton, Elkan & Narayanaswamy(2014), P2 rank-based Gaussian dependence는 Liu, Lafferty & Wasserman(2009) 및 Cai & Zhang(2018), covariate-dependent copula는 Patton(2006)과 Vatter & Nagler(2018), P3 masked representation은 Nie et al.(2023)과 Dong et al.(2023)을 참고했다. 반복 validation selection의 편향은 Cawley & Talbot(2010)의 경고에 따라 해석했다. 상세 URL과 적용 한계는 `claim-source-ledger.md`에 기록돼 있다.

최종 원칙은 단순하다. 문헌상 가능한 방법, training-only 구조 support, frozen historical 성능, 공식 generalization을 서로 분리한다. 이번 사이클은 앞의 세 단계까지만 수행했고, 그 결과 어느 문제에서도 공식 제출 후보가 되지 못했다.

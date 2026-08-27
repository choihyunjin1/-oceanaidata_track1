# P1·P2·P3 승격 기준 Deep Research v1

작성일: 2026-08-27 KST  
상태: 연구 완료, 운영 규격 봉인, 제출·업로드 없음  
범위: 로컬 연구 결과의 과학적 승격, 공식 Public 탐색, 최종 Private 모델 선택 기준

## 결론

현재 가장 큰 구조적 결함은 모델 자체보다 **서로 다른 세 가지 결정을 모두 “승격”이라고 부른 것**이다.

1. 로컬 연구면에서 유망한가?
2. 공식 Public 점수상 지금 최고인가?
3. 보이지 않는 Private에서도 최종 모델로 채택할 근거가 충분한가?

이 셋은 동일하지 않다. 기존 P1·P2·P3 로컬 OOF 면은 반복적인 후보 탐색에 이미 노출됐고, 공식 Public도 같은 숨은 표본을 반복 질의한다. 따라서 작은 로컬 CI 통과나 Public 최고 갱신만으로는 구조적 일반화를 주장할 수 없다. 유한 검증 기준 자체를 최적화하면 선택 편향이 생길 수 있다는 결과와, 반복적인 leaderboard 피드백이 holdout 과적합을 유발한다는 연구가 이를 직접 뒷받침한다([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [Blum & Hardt, 2015](https://proceedings.mlr.press/v37/blum15.html)).

따라서 새 기준은 한 줄짜리 pass/fail이 아니라 다음 두 축을 동시에 기록한다.

- **과학적 증거 상태**: `QA_BLOCKED`, `RESEARCH_ONLY`, `CONFIRM_MORE`, `LOCAL_CONFIRMED`, `LOCAL_CONFIRMED_STRONG`, `FUTILITY_STOP`, `REJECT_HARM`, `EQUIVALENT`
- **대회 행동 상태**: `NO_OFFICIAL_ACTION`, `OFFICIAL_PROBE_ELIGIBLE`, `PUBLIC_BEST_ONLY`, `PRIVATE_READY`

Public 점수가 좋아지면 `PUBLIC_BEST_ONLY`가 될 수 있다. 그러나 fresh local confirmation과 재현성까지 갖추기 전에는 `PRIVATE_READY`가 아니다. 이 분리는 소수점 단위의 실제 Public 개선을 버리지 않으면서도, 그 개선을 과학적 확인으로 과장하지 않게 한다.

현재 결과를 새 기준으로 재분류하면 다음과 같다.

| 문제·후보군 | 과학적 상태 | 연구 처분 / 대회 행동 | 판정 |
|---|---|---|---|
| P1 endpoint-unanimity topology bridge | `RESEARCH_ONLY` | `STOP_EXACT_RULE_NO_HARM_CLAIM` / `NO_OFFICIAL_ACTION` | 노출 surface에서 효과 `-0.001454 F1`, 1/3 fold 개선. CI가 없어 통계적 harm/futility까지 주장하지 않되 이 정확한 규칙은 중단 |
| P2 universal density soft-loss | `RESEARCH_ONLY` | `STOP_EXACT_UNIVERSAL_PENALTY_NO_FRESH_FUTILITY_CLAIM` / `NO_OFFICIAL_ACTION` | prior-exposed internal confirmation-style screen의 benefit `-0.000678℃`, 90% CI `[-0.002019,+0.000747]`; 추가 미세조정은 중단하되 pristine holdout이 아니므로 formal futility는 주장하지 않음 |
| P3 lead-continuous correction | `RESEARCH_ONLY` | `FRESH_CONFIRMATION_PRIORITY` / 조건부 `OFFICIAL_PROBE_ELIGIBLE` | exposed surface의 active benefit `+0.004188m`, 90% CI `[-0.001585,+0.010129]`; 저장 CI도 anchor-day IID resampling이라 새 contiguous day/episode block 기준을 충족하지 않음. fresh 78h-separated confirmation 전에는 G2의 `CONFIRM_MORE`를 부여하지 않음 |

현재 동결된 Round E 9개 파일은 P1·P2·P3 각각 세 번의 독립 확인이 아니다. 문제별로 하나의 상관된 factorial/axis family를 세 점으로 식별하는 **공식 정보획득 배치**다. 점수가 관측되면 Public 챔피언과 기전 단서를 만들 수 있지만, 그것만으로 `LOCAL_CONFIRMED`나 `PRIVATE_READY`가 되지는 않는다.

## 연구 질문과 답

### Q1. 작은 개선도 승격해야 하는가?

공식 Public 챔피언 교체에는 그럴 수 있다. 점수 반올림·계보·scorer 오차보다 큰 결정적 개선이면 수치상 최고 파일을 바꾸는 것은 합리적이다. 반면 과학적 모델 승격에는 “0보다 큼”만으로 부족하다. 후보를 보기 전에 정한 최소 가치 개선폭 `δ_gain`과 불확실성을 함께 써야 한다. p-value만으로 효과의 중요성을 판단할 수 없다는 ASA 원칙과, 비유의가 곧 동등함을 뜻하지 않는 SESOI/equivalence 원칙이 이 구분을 지지한다([ASA Statement, 2016](https://doi.org/10.1080/00031305.2016.1154108), [Lakens et al., 2018](https://doi.org/10.1177/2515245918770963)).

### Q2. 모든 fold·lead·layer가 좋아야 안전한가?

아니다. 2~3개 fold의 부호를 독립 표본처럼 취급하면 검정력이 지나치게 낮고, 한 subgroup에서 유의하고 다른 subgroup에서 유의하지 않다는 사실만으로 효과 차이를 주장할 수도 없다([Gelman & Stern, 2006](https://doi.org/10.1198/000313006X152649), [Altman & Bland, 2003](https://doi.org/10.1136/bmj.326.7382.219)).

따라서 aggregate primary 효과가 주 판단이며, fold 일관성은 보조 진단으로 내린다. hard veto는 다음에만 둔다.

- 누수·키·계보·시간 중복 등 integrity 실패
- 사전 지정한 운영상 critical slice의 동시 비열등성 경계 실패
- aggregate에서 실용적 harm가 충분히 지지됨

### Q3. 기존 90% bootstrap CI는 충분한가?

오직 **한 개의 사전등록 primary 후보를 fresh confirmation surface에서 평가**할 때 one-sided `α=0.05`와 대응하는 중앙 90% CI로 해석할 수 있다. 반복 노출 OOF에서 여러 후보를 보고 가장 좋은 것만 고른 뒤 같은 CI를 쓰면 selection multiplicity가 포함되지 않는다. K-fold CV 분산에는 보편적인 불편 추정량이 없고 겹친 학습집합의 상관을 무시하면 분산을 과소평가할 수 있다([Bengio & Grandvalet, 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html)).

세 후보 모두 우월성을 주장한다면 Romano–Wolf stepdown max-T로 FWER 0.05를 제어한다([Romano & Wolf, 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)). boundary null로 recenter한 후보별 statistic을 가능하면 studentize하고, 모든 후보에 동일한 dependent-block resample index를 공유하며, stepdown adjusted p-value 또는 critical value를 계산한다. 같은 raw percentile replicate만 공유한 것은 Romano–Wolf가 아니다. 구현이 불가능하면 후보당 one-sided `α=0.0167`, 즉 중앙 96.67% CI를 쓰는 Bonferroni 대안을 사용한다. critical slice는 Holm-adjusted one-sided NI test 또는 그 inversion simultaneous lower bound를 사용한다. 한 후보만 confirmatory이고 나머지 둘을 diagnostic으로 봉인하면 primary에는 one-sided `α=0.05`를 유지할 수 있으나, diagnostic 두 개는 승격 근거로 재해석할 수 없다.

### Q4. 같은 leaderboard를 다음 날 다시 쓰면 fresh test가 되는가?

아니다. 날짜와 quota가 초기화돼도 숨은 Public 표본은 그대로다. 이전 점수로 다음 후보를 만들면 후보와 Public holdout이 적응적으로 종속된다. 같은 Public 점수의 반복 질의는 새 독립 표본이 아니므로 confidence sequence나 alpha-spending으로 이 문제를 해결할 수도 없다. 해당 방법은 새 관측이 누적되는 상황을 전제로 한다([Howard et al., 2021](https://doi.org/10.1214/20-AOS1991)).

## 내부 증거 감사

### 반복 노출 정도

- P1: 421,032행 OOF에 outer-result 13회, 10개 family, 최소 5,265 candidate-fold 평가, virgin tail 0.
- P2: 동일 세 block에서 최소 18 generation, 17 result, 반복 grid/stack/model 선택. fitted stack optimism gap `0.029846℃`.
- P3: 같은 key의 OOF artifact 최소 10개, 같은 1,092행 RMSE 문서 30개, virgin holdout 없음.

따라서 이 표면들은 모두 `development surface`다. 여기서 얻은 nominal CI는 후보 발견에는 유용하지만, 독립 확인 없이 `LOCAL_CONFIRMED`가 될 수 없다.

### local↔official 운송성

관측된 공식 contrast 13개에서 부호 일치는 6/13=`46.2%`였고 Wilson 95% CI는 `[23.2%,70.9%]`다. 아직 관측되지 않은 P3 C1 additivity prediction까지 diagnostic row로 포함하면 6/14=`42.9%`, CI `[21.4%,67.4%]`다. 비교 가능성이 높은 5개만 남겨도 3/5=`60%`, CI `[23.1%,88.2%]`다. 어느 집계도 표본이 적고 후보가 서로 상관되어 있으므로 전 문제를 아우르는 scalar calibration을 정당화하지 않는다.

- P1은 family에 따라 부호가 맞거나 뒤집힌다. Router vs B는 local `+0.002230`, official `+0.024163`으로 방향은 맞지만 크기가 10.84배다.
- P2는 official all-row quadratic score algebra가 매우 정확하다. global 예상 `0.537237736`, 관측 `0.537238`, 오차 `2.64×10⁻⁷`. 그러나 local layer ranking은 L2에서 부호가 뒤집혔고 L4 공식 효과는 local의 12.76배였다.
- P3 현재 correction family는 A/B/reverse-long 모두 local과 official 방향이 뒤집혔다. 현 local analogue를 scalar selector로 쓰면 안 된다.

결론은 “로컬을 버린다”가 아니다. 로컬은 leakage-free 메커니즘·비열등성·재현성을 검증하고, Public은 hidden mixture에서의 실제 점수와 transport를 측정한다. 두 표면의 역할을 분리해야 한다.

## 새 승격 상태기계

모든 문제에서 benefit을 클수록 좋게 통일한다.

- P1: `b = F1_candidate − F1_incumbent`
- P2/P3: `b = RMSE_incumbent − RMSE_candidate`

후보 결과를 보기 전에 `δ_gain`(최소 가치 개선), `δ_harm,s`(critical slice 허용 악화), 필요하면 `δ_eq`(실용적 동등 범위)를 봉인한다.

### 과학적 증거 축

1. `QA_BLOCKED`
   - 키·행·순서·해시·계보·누수·시간 격리·재현성 중 하나라도 실패.
   - 성능 실패가 아니라 측정 불가능 상태다.

2. `RESEARCH_ONLY`
   - 노출된 discovery surface에서 얻은 유리하거나 불리한 모든 결과.
   - 연구 우선순위·family 중단·공식 probe 가치는 정할 수 있으나 confirmatory population claim은 금지.

3. `LOCAL_CONFIRMED_STRONG`
   - fresh locked confirmation.
   - `b_hat ≥ δ_gain`이고 paired dependent-bootstrap 90% CI `[L,U]`에서 `L ≥ δ_gain`.
   - integrity와 simultaneous critical-slice noninferiority 통과.

4. `LOCAL_CONFIRMED`
   - fresh locked confirmation.
   - `b_hat ≥ δ_gain`, `L > 0`, integrity와 slice noninferiority 통과.
   - 개선 존재는 지지하지만 최소 개선폭 전체를 보장하진 않는다.

5. `CONFIRM_MORE`
   - 위 승격 실패이나 `U ≥ δ_gain`.
   - 의미 있는 효과 가능성이 남아 있어 새 독립 기간을 확보할 가치가 있음.

6. `FUTILITY_STOP`
   - fresh confirmation에서 `U < δ_gain`.
   - 정확히 그 후보·가중치·보편적 메커니즘에 대해 최소 가치 개선을 배제한다. 새 계절 상호작용 모델 등은 새 hypothesis로 등록할 수 있다.

7. `REJECT_HARM`
   - aggregate 또는 진짜 safety/SLA slice에서 practical harm가 지지됨.
   - benefit convention에서 `U < −δ_harm`.

8. `EQUIVALENT`
   - CI 전체가 사전고정 `[-δ_eq,+δ_eq]` 안에 있음.
   - 단순 non-significant와 다르다. 더 단순·저렴한 후보에만 교체 근거가 될 수 있다.

상태 판정 순서는 `QA_BLOCKED → RESEARCH_ONLY → REJECT_HARM → EQUIVALENT(사전등록 primary objective일 때만) → LOCAL_CONFIRMED_STRONG → LOCAL_CONFIRMED → FUTILITY_STOP → CONFIRM_MORE`로 고정한다. superiority 실패 후 equivalence를 사후 선언하지 않는다. critical slice `s`의 비열등성 통과는 simultaneous `L_s > -δ_harm,s`, practical harm 지지는 simultaneous `U_s < -δ_harm,s`로 정의한다. `L_s ≤ -δ_harm,s ≤ U_s`이면 harm가 아니라 `NI_NOT_ESTABLISHED`이며 승격만 보류한다.

### 대회 행동 축

1. `NO_OFFICIAL_ACTION`
   - integrity 실패, exact-rule stop, futility, near-duplicate, 또는 결과가 어떤 구간에 와도 다음 행동이 변하지 않음.

2. `OFFICIAL_PROBE_ELIGIBLE`
   - lineage/scorer가 봉인됐고, 결과 구간별 다음 행동이 사전등록됐으며, 구조적 불확실성을 줄이거나 realistic exploit gain이 있음.
   - 이는 제출 가치 판정이지 모델 승격이 아니다.

3. `PUBLIC_BEST_ONLY`
   - 공식 점수가 incumbent보다 score 표시·재현 허용오차를 넘어 결정적으로 좋음.
   - Public 최고 파일은 교체할 수 있지만 Private 일반화 주장은 금지.

4. `PRIVATE_READY`
   - `LOCAL_CONFIRMED` 이상, 재현성 통과, 공식 방향이 일치하고 critical harm가 없음.
   - Public만 개선했거나 local confirmation이 노출된 표면뿐이면 금지.

## 단계별 gate

### G0 — 무결성, fail-closed

- 공식 metric·키·schema·행·순서·hash 일치
- 시간/episode 누수 없음, purge/embargo 충족
- incumbent와 candidate가 동일한 truth·key·evaluation population을 사용
- prereg·runner·manifest가 동일한 canonical gate spec SHA-256을 참조
- score를 보기 전 후보, split, metric, 독립단위, `δ_gain`, critical slice, 다음 행동을 봉인
- family ledger에 local candidate 수, official physical submission 수, adaptive family query 수를 모두 누적

G0 실패 시 성능 수치가 좋아도 `QA_BLOCKED`다.

### G1 — discovery

- 한 hypothesis family당 원칙적으로 3개 이하 설정을 노출된 surface에서 탐색
- paired cluster 효과와 구간을 계산하되 `RESEARCH_ONLY` 이상의 과학적 해석 금지
- 결과 기반 slice 선택, threshold 변경, metric 교체 금지
- HPO·threshold·model 선택은 inner chronological blocked folds에서만 하고, development 효과 요약은 inner 선택에 쓰지 않은 outer rolling-origin folds에서 계산한다. repeated folds/seeds는 상관된 robustness sensitivity이며 독립 confirmation 수로 세지 않는다.

### G2 — fresh local confirmation

- family당 한 locked winner만 들어감
- 모델/threshold/slice/HPO에 한 번도 사용되지 않은 chronological/episode-disjoint surface
- primary aggregate 하나만 지정; 나머지는 sensitivity
- paired dependent bootstrap으로 metric 자체를 replicate마다 다시 계산
- critical slice는 simultaneous one-sided noninferiority bound로 보호
- equal-tailed percentile 또는 test inversion과 동치인 중앙 90% interval의 해당 tail만 단일 사전등록 one-sided `α=0.05` 경계판정에 사용한다. 임의 BCa/studentized interval에 이 동치를 자동 가정하지 않는다.

한 번 연 confirmation surface는 다음 cycle부터 discovery surface로 강등한다.

### G2a — dependent resampling 실행 규격

primary resampler와 block length는 target access 전에 코드·버전·SHA-256으로 봉인한다. 기본은 circular stationary bootstrap과 Patton–Politis–White(2009)의 corrected Politis–White selector다([Patton, Politis & White, 2009](https://doi.org/10.1080/07474930802459016)). selector input, `ceil` 규칙, 최소·최대 clamp, circular-boundary convention, seed, replicate 수를 runner에 고정한다.

- stationary bootstrap은 `N_unique_units < 4×B_min`이면 `QA_BLOCKED_INSUFFICIENT_DEPENDENT_UNITS`로 fail-closed한다. 그 외에는 `B_max=floor(N_unique_units/4)`로 고정해 expected block count가 최소 4가 되게 한다. 이 clamp는 대회 운영 규칙이며 문헌 정리가 아니다.
- P1: day-level paired micro-F1 influence input을 코드로 봉인하고 `B_min=4`, `B_primary=min(B_max,max(4,ceil(B_hat)))`. bootstrap replicate에서는 influence 근사가 아니라 TP/FP/FN을 다시 합쳐 정확한 micro-F1을 계산한다.
- P2: primary temperature benefit `b=RMSE_inc−RMSE_cand`의 code-pinned day-level influence series를 selector input으로 사용한다. L2–L4를 row-count weighted pooled하며, T/S는 함께 resample하되 salinity 값은 selector와 primary statistic에 섞지 않는다. `B_min=1`, `B_primary=min(B_max,max(1,ceil(B_hat)))`.
- P3: 사전 봉인한 storm episode ID가 있고 distinct episode cluster가 최소 8개면 whole-episode cluster bootstrap이 primary다. 8개 미만이면 episode primary는 fail-closed한다. episode ID가 없으면 six-lead pooled RMSE benefit의 anchor-day influence series를 사용하고 각 origin의 6개 lead를 유지한다. 이 fallback은 `B_min=1`, distinct anchor-day 최소 8개, `B_primary=min(B_max,max(1,ceil(B_hat)))`이며 개별 day IID resampling은 금지한다. 최소 8은 본 연구의 운영상 하한이지 충분한 power 보장이 아니다.
- seed와 replicate 수는 결과 전에 봉인하며 기본 replicate 수는 10,000이다.
- P1 4/7/14일, P2 3/7/14일은 sensitivity 전용이다. 유리한 길이를 사후 primary로 고르지 않는다. primary가 통과해도 사전 sensitivity가 방향 또는 NI를 뒤집으면 `SENSITIVITY_UNSTABLE`로 기록하고 추가 자료 전 `PRIVATE_READY`를 금지하되 `REJECT_HARM`으로 부르지 않는다.

### G3 — official probe

- 업로드 전 server quota를 로그인 상태에서 확인
- 한 문제의 A/B/C 파일·해시·순서·score interval별 행동을 첫 점수 전에 동결
- A guard 외에는 중간 점수로 B/C 내용을 수정하지 않음
- 동일 축의 세 점은 세 independent confirmation이 아니라 한 family query batch로 기록
- `adaptive_family_queries=1`은 exposure/운영 원장 단위일 뿐 FWER의 hypothesis 수 `m`이 아니다. A/B/C가 모두 superiority claim이면 같은 batch여도 `m=3`이다.
- official per-row loss가 없으므로 Public 점수 차이에 population CI를 붙이지 않음

### G4 — Public 챔피언

- scorer/lineage guard 통과
- incumbent 대비 표시 반올림과 replay tolerance를 넘는 결정적 개선
- 수치상 Public best는 즉시 기록할 수 있으나 상태는 `PUBLIC_BEST_ONLY`

### G5 — Private 최종 승격

- `LOCAL_CONFIRMED` 또는 `LOCAL_CONFIRMED_STRONG`
- full pipeline 재현성·해시·계보 통과
- 공식 방향 일치 또는 score tolerance 내 비열등
- 사전 critical slices에서 simultaneous harm guard 통과
- 구조 복잡도가 증가하면 개선이 complexity/운영 최소가치도 넘음

## 문제별 구현 규격

### P1 — 비가법 F1

- bootstrap 단위: 모든 station/layer를 같은 날짜 묶음으로 유지한 joint KST-day contiguous block.
- 각 replicate에서 TP/FP/FN을 다시 합쳐 micro-F1을 재계산한다. 행별 F1 차 평균은 금지.
- 이벤트 길이가 약 3.6일까지 이어질 수 있으므로 최소 4일 block, corrected automatic selector와 4/7/14일 사전 sensitivity를 권장한다.
- provisional `δ_gain = +0.0015 F1`은 현재 exact postprocess family의 기존 prereg를 유지한다. 새 backbone처럼 비용과 자유도가 큰 family는 결과 전에 더 큰 family-specific `δ_gain`을 봉인한다.
- sparse cell은 support threshold와 동시 noninferiority를 사용한다. “모든 cell/fold 개선”은 hard veto가 아니다.
- 공식 G-only/I-only/GI는 factorial contrast다. F1의 비선형성 때문에 행 단위 additive causal effect로 해석하지 않는다.

### P2 — all-row integrated RMSE

- bootstrap 단위: L2–L4와 T/S를 함께 보존한 joint KST-day contiguous block.
- replicate마다 전체 pooled RMSE를 직접 재계산한다.
- block selector와 3/7/14일 sensitivity; same-season 기간을 primary로, 다른 계절은 predeclared transport sensitivity로 둔다.
- provisional `δ_gain = 0.001℃` benefit은 density confirmation의 기존 prereg를 소급 변경 없이 유지한다.
- exact official MSE quadratic algebra는 scorer/lineage gate로 사용하되, local layer 선택의 일반화 증거로 사용하지 않는다.
- universal density penalty의 과학적 상태는 `RESEARCH_ONLY`, 연구 처분은 `STOP_EXACT_UNIVERSAL_PENALTY_NO_FRESH_FUTILITY_CLAIM`이다. 두 block은 첫 screen과 disjoint하고 결과를 보지 않고 선택됐지만 과거 내부 artifact에 노출된 기간이다. 저장된 paired-day CI는 row-level OOF가 남지 않아 독립 QA가 재생성하지 못했고, aggregate RMSE와 verdict 산술만 재현됐다. 또한 기존 CI는 calendar-day IID resampling이어서 새 contiguous block 기준을 충족하지 않는다. 따라서 formal `FUTILITY_STOP`으로 과장하지 않는다. 계절·regime 조건부 penalty를 시도하려면 새 family, 새 rationale, fresh confirmation을 요구한다.

### P3 — forecast origin/episode 의존 RMSE

- 한 forecast origin의 6개 lead를 절대 분리하지 않는다.
- 같은 UTC anchor-day, 가능하면 같은 storm episode를 함께 묶은 contiguous block bootstrap을 사용한다.
- globally 78h-separated, episode-disjoint chronological surface가 fresh confirmation의 최소조건이다.
- 유일한 efficacy primary는 active subset benefit이며 provisional `δ_gain = 0.005m`다. 과거 overall `0.003m`은 legacy effect reference인 descriptive sensitivity로만 보존하고 promote/veto 권한을 주지 않는다. hard overall guard가 필요하면 별도의 `δ_harm,overall`을 결과 전에 정하고 simultaneous `L_overall > -δ_harm,overall` 비열등성으로 정의한다.
- 모든 lead/fold 개선은 supportive일 뿐 CI를 대신하지 않는다.
- 현재 lead-continuous 결과의 CI half-width와 point effect를 단순 root-n으로 진단하면 CI lower bound를 0 위로 올리는 데 약 1.96배 effective anchor-day block이 필요하다. 기존 active 96일 기준 약 190개 independent-equivalent day는 탐색적 출발점일 뿐이다. 현재 point benefit `0.004188m`가 `δ_gain=0.005m`보다 작으므로 같은 효과가 유지되면 표본수 증가만으로 `LOCAL_CONFIRMED` gate를 통과하지 못한다. 정식 power target은 `δ_gain` 이상의 설계효과와 block dependence를 가정해 다시 계산해야 한다.
- 현재 저장 CI는 UTC anchor-day를 IID로 재표집해 연속 day/episode dependence를 보존하지 않는다. 따라서 방향성 진단에는 쓰되 새 G2 불확실성 기준을 통과한 것으로 취급하지 않는다.
- 현 local analogue와 official correction 방향이 반복 역전됐으므로 Public axis 결과는 transport 진단으로만 사용한다.

## 하루 3회 공식 제출의 정보가치 설계

정확한 현재 quota는 공개 비로그인 API에서 확인되지 않았다. 공식 live client는 문제별 quota endpoint, server `remaining/limit`, 제출 직후 Public metric, 문제별 최고 Public 합산, 마감 후 Private 확정을 사용한다. “문제당 하루 3회”는 사용자 진술과 2026-08-26 인증 UI 관측으로 지지되므로 실제 업로드 직전에 로그인 상태에서 재확인해야 한다.

권장 A/B/C 역할은 다음과 같다.

1. **A — calibration/lineage sentinel**
   - 점수를 사전에 거의 정확히 예측할 수 있는 항등식·known mixture·무변화 guard.
   - mismatch면 후속 해석을 중단한다. A 점수로 B/C를 성능 최적화하지 않는다.

2. **B — mechanism ablation**
   - 한 구성요소만 격리한 orthogonal diagnostic.
   - 단독 점수로 승격 금지.

3. **C — exploitation primary**
   - fresh local gate를 통과했거나, local transport가 약한 문제에서는 가장 높은 사전 정보가치를 가진 locked candidate.
   - 그날의 유일한 confirmatory claim 후보.

lineage가 이미 충분히 검증된 문제는 A 대신 두 번째 독립 구조축을 쓸 수 있다. 후보 prediction-change vector의 pairwise cosine 절댓값이 0.95를 넘거나 delta matrix의 effective rank가 약 1이면, 사전등록한 curvature/interaction 식별식이 없는 한 near-duplicate로 간주한다. 이 0.95는 문헌 정리가 아니라 본 대회의 운영상 중복 방지 기준이다.

quota는 다음 날 복구되지만 Public holdout 노출은 누적된다. 결과가 어느 구간에 와도 다음 행동이 같다면 제출하지 않고, 독립적인 구조축 후보를 찾는다. 반대로 한 점수가 계보 오류를 막거나 내일의 후보 family를 갈라놓는다면 leaderboard 최고를 갱신하지 못해도 정보가치가 있다. 이 탐색–활용 구분은 정보비율 관점의 실무적 유추이며, 각 arm에서 독립 확률표본을 받는 bandit으로 대회를 오해하지 않는다([Russo & Van Roy, 2018](https://doi.org/10.1287/opre.2017.1663)).

## 기존 정책과의 관계

다음 과거 규칙은 이 문서가 **2026-08-27 이후 새로 시작하는 내부 연구·승격 cycle에 한해** 대체한다. 원본은 감사용으로 보존하며, 과거 실험의 사전등록 판정, 이미 봉인된 Round E 파일·순서·식별식, 주최측 규칙을 소급 변경하지 않는다.

- `artifacts/validation_system_audit_20260822/cross_problem_policy.json`
  - 첫 공식 점수 후 후보 pool 영구 폐쇄
  - P1 `0.02 F1`, P2 `0.012101℃`, P3 `0.028792m`의 단일 switch margin
- `20260825_LOCAL_OFFICIAL_CALIBRATION_LEDGER.json`
  - local과 official 방향이 일치할 때만 모든 종류의 승격 허용

과거 큰 margin은 폐기하지 않고 **Private-risk strong margin**으로만 참고할 수 있다. Public 수치상 최고 교체 threshold로 강제하면 실제 Round D P2 `0.004549℃`, P3 `0.007999m` 개선까지 거절하는 모순이 생긴다. 반대로 Public 개선 하나로 Private-ready를 선언해서도 안 된다.

## Round E 사전 해석 규칙

동결 manifest SHA-256: `7dd80d6288cd957192055916627b6bd31778565defb60f56c9baf078c8d487bc`  
현재 상태: `FROZEN_READY_NOT_UPLOADED`, uploads `0`

- P1 G-only/I-only/GI: 한 factorial family. 세 독립 복제 아님.
- P2 U/E/F: 한 official axis의 exploit + envelope/PAVA ablation. 세 독립 복제 아님.
- P3 α=-2/α=-4/subset: 한 reverse-axis family. 세 독립 복제 아님.

관측 후 허용되는 것은 다음뿐이다.

- score identity/lineage guard 판정
- family 내 Public best 기록
- 사전 식별식에 따른 main effect/interaction/curvature 방향 기록
- 다음 날 새로운 local hypothesis 우선순위 결정

금지되는 것은 다음이다.

- 세 점을 independent replication으로 세기
- Public best를 곧바로 `LOCAL_CONFIRMED` 또는 `PRIVATE_READY`로 부르기
- score를 본 뒤 같은 날 파일·가중치·threshold 변경
- 실패한 diagnostic을 결과 후 confirmatory로 재명명

## 공식 운영상 미확인 사항

2026-08-27 비로그인 상태에서 leaderboard와 quota API는 401이어서 현재 팀 순위·잔여 횟수·server limit을 독립 확인하지 못했다. live client bundle은 다음 구조를 확인시킨다.

- 문제별 quota 조회와 `remaining/limit`
- 제출 직후 Public metric/score
- 문제별 Public 최고 점수 합산, 약 20초 갱신
- 마감 뒤 Private 점수로 최종 확정
- 모델 최종 제출 후 해당 문제의 추가 답안 업로드 잠금

내부 메모가 기록한 과거 참가자 공지와 현재 공개 홈페이지 일정이 충돌하는 것으로 보고되어 있다. 본 QA에서는 인증 공지 원문을 독립 검증하지 못했으므로 최종 모델 확정 직전에 로그인 공지를 재확인해야 한다. 본 연구는 어떤 업로드나 최종 모델 확정도 수행하지 않았다. 공식 test/sample 내용 비열람은 실행 scope receipt이며 시스템 수준 file-access 감사로그로 독립 증명한 것은 아니다. 독립 QA는 Round E manifest와 9개 candidate byte hash가 모두 동결값과 일치함을 확인했다.

## 한계

- 문헌은 보편적인 `δ_gain` 수치를 주지 않는다. score 분해능, pipeline seed 변동, local↔official transport 잔차, 복잡도 비용을 결합해 family별로 결과 전에 정해야 한다.
- block bootstrap은 dependence를 다루지만 distribution shift를 해결하지 않는다. 여러 chronological period와 rolling origin이 필요하다.
- official per-row loss가 없으면 Public score 차이에 모집단 불확실성 CI를 만들 수 없다.
- local↔official 관측 contrast는 13개이며, 미관측 P3 C1 예측 진단행을 포함한 calibration row는 14개다. 서로 상관되어 global calibration을 학습하기에 부족하다.
- pairwise cosine 0.95 중복 기준과 슬롯 A/B/C는 대회 운영 규칙이지 통계 정리의 직접 결과가 아니다.

## 바로 적용할 다음 단계

1. 이 보고서와 `promotion_gate_v1.json`을 다음 cycle의 governing spec으로 사용한다.
2. 새 실험은 시작 전에 canonical spec SHA를 prereg·runner·manifest에 동일하게 pin한다.
3. P1은 exact bridge를 종료하고 새로운 family는 fresh chronological label reserve를 먼저 확보한다.
4. P2 universal density penalty를 종료하고, 새로 하려면 regime interaction hypothesis로 분리한다.
5. P3는 약 190 independent-equivalent active anchor-day를 CI 폭 진단의 출발점으로만 참고하고, `δ_gain≥0.005m` 설계효과와 block dependence를 넣어 정식 power를 다시 계산한 뒤 globally 78h-separated confirmation을 설계한다.
6. Round E를 실행한다면 9개를 “3개 문제 × 각 1개 family query batch”로 기록하고, Public best와 과학적 승격을 별도 열에 남긴다.
7. 업로드 직전 로그인 quota와 model-finalize lock, 최신 deadline을 다시 확인한다.

## 주요 1차 출처

- [Cawley & Talbot, model-selection overfitting, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Blum & Hardt, adaptive leaderboard overfitting, ICML 2015](https://proceedings.mlr.press/v37/blum15.html)
- [Dwork et al., reusable holdout, Science 2015](https://doi.org/10.1126/science.aaa9375)
- [Bengio & Grandvalet, k-fold variance, JMLR 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html)
- [Roberts et al., structured cross-validation, Ecography 2017](https://doi.org/10.1111/ecog.02881)
- [Cerqueira et al., time-series evaluation, Machine Learning 2020](https://doi.org/10.1007/s10994-020-05910-7)
- [Politis & Romano, stationary bootstrap, JASA 1994](https://doi.org/10.1080/01621459.1994.10476870)
- [Patton, Politis & White, corrected block-length selector, Econometric Reviews 2009](https://doi.org/10.1080/07474930802459016)
- [Diebold & Mariano, predictive accuracy, JBES 1995](https://doi.org/10.1080/07350015.1995.10524599)
- [White, Reality Check for data snooping, Econometrica 2000](https://doi.org/10.1111/1468-0262.00152)
- [Hansen, Superior Predictive Ability, JBES 2005](https://doi.org/10.1198/073500105000000063)
- [Romano & Wolf, stepdown multiple testing, Econometrica 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)
- [Lakens et al., equivalence and SESOI, AMPPS 2018](https://doi.org/10.1177/2515245918770963)
- [Howard et al., time-uniform confidence sequences, Annals of Statistics 2021](https://doi.org/10.1214/20-AOS1991)
- [Russo & Van Roy, information-directed sampling, Operations Research 2018](https://doi.org/10.1287/opre.2017.1663)

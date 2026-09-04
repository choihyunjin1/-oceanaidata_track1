# P1·P2·P3 과거 실험 새 승격기준 재정찰

상태: `COMPLETE_RETROSPECTIVE_RECLASSIFICATION`  
기준일: 2026-08-27 KST  
적용 기준: `artifacts/promotion_gate_deep_research_20260827_v1/promotion_gate_v1.json`  
성격: 과거 증거의 재해석이며, 과거 계약·결정·제출 파일을 수정하는 작업이 아니다.

## 결론

과거 P1·P2·P3 실험 중 새 기준으로 `LOCAL_CONFIRMED`, `LOCAL_CONFIRMED_STRONG`, `PRIVATE_READY`에 도달한 family는 **0개**다. 이유는 모델이 모두 나빠서가 아니라, 성능이 계산된 로컬 OOF가 반복 탐색에 노출됐고 같은 Public leaderboard도 반복 질의돼 **새 독립 확인 표본이 아니기 때문**이다. 유한 검증 기준을 반복 최적화하면 선택 편향이 생기고, 반복 leaderboard 피드백은 holdout 의존성을 만든다는 연구와 일치한다([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [Blum & Hardt, 2015](https://proceedings.mlr.press/v37/blum15.html)).

그러나 “아무것도 남지 않았다”는 결론도 틀리다.

- 현 대회상 Public 최고는 P1 Router `0.817873`, P2 Layer-4 quadratic `0.536536℃`, P3 reverse-global `0.599072m`다. 세 파일의 행동 상태는 `PUBLIC_BEST_ONLY`이며 과학적 상태는 여전히 `RESEARCH_ONLY`다.
- Round E의 P1 G/I/GI, P2 U/E/F, P3 reverse-axis curvature/subset은 각각 **세 번의 확인이 아니라 한 번의 상관된 family query**다. 현재 `OFFICIAL_PROBE_ELIGIBLE`이지 `PUBLIC_BEST_ONLY`나 `PRIVATE_READY`가 아니다.
- 과거의 많은 실패는 전체 접근법의 영구 반증이 아니다. 새 표면이 없으므로 `FUTILITY_STOP` 또는 `REJECT_HARM`으로 격상하지 않고, `STOP_EXACT_*`라는 연구 처분으로 **실제로 시험한 정확한 규칙·구조만** 닫는다.
- P3 lead-continuous는 효과 `+0.00418782m`로 새 최소 개선폭 `0.005m`에 못 미치지만 legacy CI가 넓다. 정확한 결론은 `RESEARCH_ONLY + FRESH_CONFIRMATION_PRIORITY`; 조건부 `OFFICIAL_PROBE_ELIGIBLE`다.
- P3 ERA5 context-transfer는 아직 입력 준비가 진행 중인 `RUNNING_UNEVALUATED`다. 모델 실패나 `QA_BLOCKED` 증거로 세지 않는다.

## 새 승격기준

과거의 단일 “승격/탈락” 표기를 두 축으로 분리한다.

| 축 | 상태 | 의미 |
|---|---|---|
| 과학적 증거 | `QA_BLOCKED` | 무결성·식별성 실패로 점수를 증거로 인정하지 않음 |
| 과학적 증거 | `RESEARCH_ONLY` | 노출된 discovery surface의 결과. 방향 탐색과 정확한 family 종료에는 사용 가능 |
| 과학적 증거 | `LOCAL_CONFIRMED[_STRONG]` | 한 후보를 잠근 뒤 완전히 fresh한 시계열/episode 표면에서 효과·CI·slice guard 통과 |
| 과학적 증거 | `CONFIRM_MORE`, `FUTILITY_STOP`, `REJECT_HARM`, `EQUIVALENT` | fresh G2에서만 허용되는 불확실성 기반 판정 |
| 대회 행동 | `NO_OFFICIAL_ACTION` | 공식 질의 가치 없음 |
| 대회 행동 | `OFFICIAL_PROBE_ELIGIBLE` | 사전 동결된 점수가 다음 행동을 바꿀 정보가치가 있음 |
| 대회 행동 | `PUBLIC_BEST_ONLY` | Public 점수가 재현·반올림 허용오차를 넘어 incumbent를 이김 |
| 대회 행동 | `PRIVATE_READY` | fresh local 확인, 재현성, official 방향, critical-slice 비열등을 모두 통과 |

운영 규칙은 다음과 같다.

1. 노출된 OOF에서 나온 유리·불리한 결과는 모두 `RESEARCH_ONLY`가 상한이다.
2. `STOP_EXACT_FAMILY`는 연구 자원 배분 결정이지 모집단 `FUTILITY_STOP`이 아니다.
3. Public 최고 갱신은 정당한 대회 행동이지만 Private 일반화 확인과 분리한다.
4. G2는 후보 1개, 미사용 chronological/episode-disjoint 표면, 사전 봉인된 최소 개선폭, contiguous/episode dependent bootstrap, simultaneous critical-slice noninferiority를 요구한다.
5. 세 후보의 우월성을 동시에 주장하면 Romano–Wolf stepdown 또는 Bonferroni가 필요하다. Romano–Wolf는 후보 간 의존을 이용하면서 familywise error를 통제하도록 설계됐다([Romano & Wolf, 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)).
6. workflow 상태(`DRAFT`, `RUNNING_UNEVALUATED`, `NO_SCIENTIFIC_RESULT`)는 증거 상태와 별도 기록한다.

## 공식 질의 계보와 중복 제거

확인된 공식 metadata 기준 물리 제출은 총 18개다: 2026-08-25 original/A/B의 9개와 2026-08-26 Round D의 9개다. Round C에 포장된 P1 Router/Intersection/Union과 P2 global 파일은 Round D와 SHA-256이 같고 Round C에서는 업로드 0이므로 별도 제출이나 별도 확인으로 세지 않았다.

문제별로 original을 baseline으로 두면 A, B, Round D가 세 번의 adaptive family query다. Round D 안의 세 파일도 같은 hidden Public 표면을 공유하므로 독립 확인은 0개다. Round E는 9개 파일이 동결됐지만 감사 시점 upload 0이며 문제별 한 batch, 전체 세 family query로만 기록한다.

| 문제 | original | 2026-08-25 최고 | Round D 최고 | 현 행동 상태 |
|---|---:|---:|---:|---|
| P1 F1 ↑ | 0.790709 | B 0.793710 | Router **0.817873** | `PUBLIC_BEST_ONLY` |
| P2 RMSE ℃ ↓ | 0.541085 | original 0.541085 | L4 **0.536536** | `PUBLIC_BEST_ONLY` |
| P3 RMSE m ↓ | 0.607071 | original 0.607071 | reverse-global **0.599072** | `PUBLIC_BEST_ONLY` |

관측 공식 contrast 13개에서 local–official 부호 일치는 6/13=`46.2%`였다. 아직 미관측인 P3 additivity diagnostic까지 넣으면 6/14다. 서로 다른 family와 표본이 섞였고 P1 Router는 official/local 효과비가 10.84배, P2 L4는 12.76배이며 P3 reverse family는 방향이 모두 뒤집혔다. 따라서 전 문제를 아우르는 보정식은 만들지 않는다.

## P1 재분류

2026-08-22 validation audit 시점에 이미 P1의 421,032행 OOF에는 **최소** 13회 outer-result, 10개 family, 5,265 candidate-fold 평가가 누적됐고 virgin local tail은 0이었다. 그 이후 확인된 family까지는 아래 17개 material-family ledger에 추가했으며, 이 확장은 freshness를 회복시키지 않는다. 776,706행 전체 train도 deployment fit에 사용돼 retrospective fresh window가 없다.

| ID | family | 핵심 역사 증거 | 새 증거 / 행동 | 연구 처분 |
|---|---|---|---|---|
| P1-F00 | offline tabular screen | XGB 0.860371 local, original Public 0.790709 | `RESEARCH_ONLY` / historical Public, superseded | XGB를 lineage anchor로만 유지 |
| P1-F01 | TCN·Patch Transformer | 0.767582 / 0.799755, XGB보다 열위 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact architectures 종료 |
| P1-F02 | block inpaint | benefit +0.002591, CI가 0 교차, worst slice −0.058874 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact rule 종료, harm 모집단 주장 금지 |
| P1-F03 | target-masked quantile | benefit −0.630823 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact quantile rule 종료 |
| P1-F04 | IORS external transfer | point residual benefit −0.063301; 외부 profile은 target label 부재 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | point residual 종료, profile은 domain diagnostic |
| P1-F05 | Round A causal event rescue | local +0.000571~+0.002087, Public −0.004564 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact Round A 종료 |
| P1-F06 | Round B event-day balanced | local +0.004186, Public +0.003001 | `RESEARCH_ONLY` / historical Public, superseded | backbone·factorial anchor로 유지 |
| P1-F07 | sequence successor cycles | v2/v3/v4 큰 악화, v5 exact no-op | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 시험한 exact architectures 종료 |
| P1-F08 | synthetic event injection | pooled −0.005692, folds 불안정 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact generator·threshold 종료 |
| P1-F09 | semimarkov·long-event residual | inner-only 미세 이득, outer 0 또는 rescued rows 0 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact decoder 종료 |
| P1-F10 | Round D disagreement factorial | Router Public +0.024163; Intersection +0.009218; Union −0.011404 vs B | Router `PUBLIC_BEST_ONLY`; 모두 science `RESEARCH_ONLY` | Router 유지, Union 종료, Intersection mechanism 기록 |
| P1-F11 | topology bridge | benefit −0.00145376, 1/3 folds | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | endpoint-unanimity exact rule 종료 |
| P1-F12 | Round E G/I/GI | local vs B +0.000848/+0.000606/+0.001453, upload 0 | `RESEARCH_ONLY` / `OFFICIAL_PROBE_ELIGIBLE` | 한 factorial query로만 사용 |
| P1-F13 | target density correction | structural gate fail, 새 후보 대신 B fallback 재현 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | density correction 종료; fallback 중복 금지 |
| P1-F14 | seeded boundary completion | 유효 v2 micro −0.005419, weighted −0.004215, CI 0 교차; 초기 2회 invalid | valid 결과 `RESEARCH_ONLY`, invalid subruns `QA_BLOCKED` / `NO_OFFICIAL_ACTION` | filter-fix exact rule 종료; invalid 결과는 점수 증거에서 제외 |
| P1-F15 | dynamic peer reliability gate | micro +0.004640이나 weighted +0.000121, CI 0 교차, worst group −0.047746 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | fixed24h exact peer rule 종료 |
| P1-F16 | GORS depth invariance | weighted +0.002688이나 G-ORS −0.007941, CI 0 교차, FP-day +14.38% | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact symmetric depth mask 종료 |

기술 precheck·tombstone·science-never-ran 변형은 과학적 family 결과로 세지 않았다. 해당 실행은 `QA_BLOCKED` 또는 `NO_SCIENTIFIC_RESULT` workflow exception으로 별도 ledger에 남겼다. Boundary family의 초기 invalid 두 실행은 노출 횟수에는 포함하되, 이후 유효 filter-fix 결과와 섞어 점수 증거로 사용하지 않았다.

## P2 재분류

P2의 핵심 3-block/69,850행 또는 파생 78,156행 표면은 최소 18세대와 17개 결과에서 반복 사용됐다. 큰 surrogate 이득이 exact/full-prefix/Public에서 역전된 것이 가장 일관된 패턴이다.

| ID | family | 핵심 역사 증거 | 새 증거 / 행동 | 연구 처분 |
|---|---|---|---|---|
| P2-F00 | early router/LGBM selection | 400 rounds 0.788890, 5000 rounds 0.866540 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 400-round component만 역사 참조 |
| P2-F01 | deep models and stack | fitted +0.043076, LOBO +0.013229; optimism gap 0.029846 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | fitted CI는 selection optimism 증거 |
| P2-F02 | GBM addon/top-3 HPO | CatBoost LOBO +0.001083, CI 0 교차; tuned blend weight 0 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact addon 종료 |
| P2-F03 | soft physical extrapolation | pooled v2 +0.006050, same-season CI 0 교차; original O와 hash 동일 | `RESEARCH_ONLY` / historical Public, superseded | Public axis anchor로만 유지 |
| P2-F04 | external/physical addons | TEOS 큰 악화, tide NO_GO, NASA no-op, ERA5 극미세 harm | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 시험한 exact addons 종료 |
| P2-F05 | forward surrogate stack | local +0.073375, Public −0.172435 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 대표 transport reversal; exact family 종료 |
| P2-F06 | arch-matched curve/L4 multitask | p100 −0.138081/−0.006663; exact recipe 부재 | valid run `RESEARCH_ONLY`, exact claim `QA_BLOCKED` | exact refit recipe 없이는 lineage 주장 금지 |
| P2-F07 | matched A/B causal fallback | exact local과 Public 모두 A/B 악화; B supported rows 0 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | A stack·B fallback/causal gate 종료 |
| P2-F08 | authoritative surrogate v5 | 900/900 jobs QA PASS; p100 active candidates 모두 명확한 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 등록 exact settings terminal stop |
| P2-F09 | Round D exact Public axis | global 예측 오차 2.64e−7; L2 harm, L4 Public +0.004549 | L4 `PUBLIC_BEST_ONLY`; science `RESEARCH_ONLY` | scorer/lineage gate 통과, fresh G2는 미충족 |
| P2-F10 | universal density penalty | first +0.002812, exposed confirmation −0.000678, CI upper 0.000747 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact universal weight .10 종료; regime interaction은 새 family |
| P2-F11 | Round E U/E/F | U 예상 0.535750480, 현 L4 대비 예상 +0.000785520; upload 0 | `RESEARCH_ONLY` / `OFFICIAL_PROBE_ELIGIBLE` | 한 exploit+physics family query |
| P2-F12 | annual-cycle public anomaly transfer | p100 candidate−reference +0.535506℃, late/full/fold gates 실패 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | direct annual-transfer exact family 종료 |
| P2-F13 | terminal vertical-offset transfer | p100 +0.053032℃, 세 folds 모두 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact offset-transfer family 종료 |
| P2-F14 | public-profile median consensus | p100 +0.232160℃, folds·layers 불안정 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact median-consensus family 종료 |
| P2-F15 | OAS conditional profile | p100 +0.469219℃, 모든 fold 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact OAS conditional family 종료 |
| P2-F16 | public day-sequence analog | p100 +2.164613℃, 전 fold·layer 큰 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact day-sequence analog 종료 |
| P2-F17 | RFF public-state profile | p100 +0.572589℃, 전 fold·layer 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact RFF state-profile 종료 |
| P2-F18 | prequential residual generation | H1/H2/H3 p100 +0.019464/+0.023576/+0.019665℃; late/full/slice gates 실패 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | 세 exact prequential hypotheses 종료 |

NCR 시간복원 오류, DTW inner-window target 0, dynamic-sigmoid observability 실패는 `QA_BLOCKED`다. 위 F12–F18의 development fail-fast는 fresh 모집단 실패가 아니므로 `RESEARCH_ONLY + STOP_EXACT_FAMILY`로만 해석한다.

## P3 재분류

P3의 182/181-case OOF는 same-key OOF가 최소 10회, same-grain RMSE JSON이 30개 이상 존재하고 incumbent도 adaptive하게 형성됐다. 일부 exact181 split은 78시간·episode 분리가 양호하지만 이미 라벨을 본 표면이므로 fresh G2가 아니다.

| ID | family | 핵심 역사 증거 | 새 증거 / 행동 | 연구 처분 |
|---|---|---|---|---|
| P3-F01 | core router/persistence shrink | incumbent 0.78016092; router 대비 +0.00658137 | `RESEARCH_ONLY` / retired Public | 재현 가능한 risk-control anchor |
| P3-F02 | corrected repeated-forward base | exact181 0.77910484, 강한 split integrity | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | canonical local benchmark |
| P3-F03 | positive shrink axis O/A/B | local +0.000444/+0.000275, Public −0.004609/−0.002275 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | positive shrink axis 종료 |
| P3-F04 | RevIN patch | candidate harm 0.00431369, case·episode CI도 불리 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact patch 종료, fresh harm 주장 금지 |
| P3-F05 | KMA transfer | outer +0.00252937 CI 0 교차; deployment 악화 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact deployment 종료, KMA 전체는 미반증 |
| P3-F06 | causal episode analog | inner +0.00592550, reused outer −0.00327201 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact analog chain 종료 |
| P3-F07 | spectral RFF | matched rerun benefit −0.04278254 | `RESEARCH_ONLY` / `NO_OFFICIAL_ACTION` | exact spectral RFF 종료 |
| P3-F08 | other structural generations | NLinear/sequence/dense72 악화; Gen6 no-op; 일부 reference mismatch | valid `RESEARCH_ONLY`, invalid `QA_BLOCKED` | 열거한 exact variants 종료 |
| P3-F09 | lead-continuous | active +0.00418782, legacy CI [−0.00158453,+0.01012909] | `RESEARCH_ONLY` / conditional `OFFICIAL_PROBE_ELIGIBLE` | fresh confirmation 최우선 |
| P3-F10 | Round D reverse Public axis | global +0.007999, 12h +0.000390, 18/24h +0.007689 official | global `PUBLIC_BEST_ONLY`; science `RESEARCH_ONLY` | Public champion·mechanism evidence로 유지 |
| P3-F11 | Round E curvature/subset | long −2/−4와 18/24 −4, upload 0 | `RESEARCH_ONLY` / `OFFICIAL_PROBE_ELIGIBLE` | 한 curvature/subset family query |
| P3-F12 | ERA5 context transfer | 03:29 KST raw 305/363, partial 0, final recovery running | evidence 없음 / `NO_OFFICIAL_ACTION` | `RUNNING_UNEVALUATED`; 고정 preflight 뒤에만 판정 |

Round D에서 기록된 P3 exact-axis guard FAIL은 scorer 불안정 증거로 쓰지 않는다. A가 모든 lead에서 O와 global midpoint가 아니므로 당시 global quadratic prediction 전제가 잘못됐다는 후속 erratum이 있다.

## 무엇을 살리고 무엇을 닫는가

### 유지

- P1 Router: 현재 Public champion. P1 Round B는 Router의 backbone/lineage anchor.
- P2 L4 quadratic: 현재 Public champion. global quadratic의 2.64e−7 재현은 scorer·lineage gate로 강하다.
- P3 reverse-global: 현재 Public champion. 18/24h가 개선의 대부분이라는 Public mechanism evidence를 보존한다.
- P2 p040 저자료 이득: full-budget 승격 근거가 아니라 새 regularization family의 가설 생성 신호.
- P3 lead-continuous: fresh G2를 설계할 우선 구조 가설.
- ERA5: 결과가 없는 진행 중 실험이므로 고정 protocol을 변경하지 않고 완료 여부를 기다린다.

### 정확히 닫음

`STOP_EXACT_*`는 각 표에 적힌 규칙·가중치·구조에만 적용한다. 대표적으로 P1 endpoint-unanimity bridge, P2 universal density penalty weight .10, P2 v5 등록 active grid, P3 positive-shrink axis, RevIN exact patch, causal analog chain을 닫는다. “딥러닝/외부자료/물리모델 전체가 무효” 같은 포괄 주장은 하지 않는다.

### 아직 확인할 가치가 있음

- P1: Round E addition factorial은 hidden support shift를 식별하는 Public query 가치가 있다. 다만 GI local point `+0.001453`은 exact-postprocess δ=`0.0015`에 약간 못 미친다.
- P2: Round E U/E/F는 expected exploit가 작지만 exact all-row algebra와 physical postprocess를 한 batch에서 분리한다.
- P3: Round E는 reverse-axis curvature와 lead subset을 식별한다. lead-continuous의 공식 probe와 중복 사용하지 않도록 사전 행동표를 먼저 정해야 한다.

## 공통 gap matrix

| 요구사항 | P1 | P2 | P3 |
|---|---|---|---|
| 완전히 fresh local surface | 없음 | 없음 | 없음; exact181은 분리됐어도 노출됨 |
| locked candidate 1개 | 과거 대부분 adaptive | 과거 대부분 adaptive | 일부 one-shot이나 표면 노출 |
| 새 dependent bootstrap | 미충족; day-block 필요 | 미충족; contiguous KST-day 필요 | 미충족; episode/contiguous anchor-day 필요 |
| 사전 δ | 새 규격 0.0015 F1 | 0.001℃ | active 0.005m |
| critical-slice simultaneous NI | 미충족 | 미충족 | 미충족 |
| local→official transport | family별 불안정 | layer ranking 역전 | reverse family 전부 방향 역전 |
| reproducibility | 주요 champion lineage 양호 | v5/official algebra 양호 | 일부 양호, 일부 mismatch superseded |
| Private-ready | 아니오 | 아니오 | 아니오 |

## 다음 실행 규칙

1. 새 모델을 만들기 전에 문제별로 **한 개의 미사용 confirmation surface**를 물리적으로 봉인한다. 현재 데이터에 virgin tail이 없다면 기존 데이터로 `LOCAL_CONFIRMED`를 만들려고 하지 않는다.
2. discovery에는 family당 기본 3 settings까지만 허용하고 inner chronological folds에서만 선택한다.
3. G2에서는 한 candidate만 잠그고 P1 joint day blocks, P2 joint KST-day blocks, P3 whole episode 또는 contiguous anchor-day blocks로 exact metric을 매 replicate 재계산한다.
4. 공식 제출은 `PUBLIC_BEST_ONLY` exploitation과 mechanism query를 구분한다. 같은 Public 점수는 다음 날 quota가 복구돼도 새 확인 표본이 아니다.
5. Round E를 실행한다면 문제별 3개 점수를 본 뒤 이름·역할·threshold를 바꾸지 않고, 사전 정의한 contrast와 다음 local 가설만 기록한다.
6. Private finalization 전에는 세 Public champion 모두 fresh local G2와 simultaneous slice guard가 없다는 위험을 명시한다.

## 감사 범위와 한계

- 공식 test/sample/submission CSV의 데이터 값은 읽지 않았다. 공식 점수·시간·SHA는 저장된 metadata JSON만 사용했다.
- 과거 파일과 Round E manifest/후보를 변경하지 않았고 업로드도 수행하지 않았다.
- 48개 material family record와 기술 예외를 family 단위로 합쳤다. technical recovery, 같은 SHA의 재포장, version-only QA 수정은 독립 scientific family로 중복 계수하지 않았다.
- 과거 legacy CI를 새 dependent-bootstrap CI로 소급 변환하지 않았다. row-level OOF가 저장되지 않은 경우에는 더욱 불가능하다.
- `PUBLIC_BEST_ONLY`는 competition decision이지 hidden Private 성능의 보증이 아니다.

## 핵심 출처

- 새 기준: `artifacts/promotion_gate_deep_research_20260827_v1/promotion_gate_v1.json`
- 공식 original/A/B: `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260825_OFFICIAL_SCORE_RECONCILIATION.json`
- 공식 Round D: `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260826_round_D_preregistered_P1x3_P2x3_P3x3/OFFICIAL_RESULTS_20260826.json`
- Round E: `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260827_round_E_preregistered_P1x3_P2x3_P3x3/SET_MANIFEST.json`
- local↔official calibration: `reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json`
- P1 exposure: `artifacts/validation_system_audit_20260822/p1.json`
- P2 exposure: `artifacts/validation_system_audit_20260822/p2.json`
- P3 exposure: `artifacts/validation_system_audit_20260822/p3.json`
- P3 ERA5 metadata receipt: `artifacts/promotion_retroaudit_20260827_v1/p3_era5_filename_process_receipt.json`
- P2 terminal v5 synthesis: `reports/final_goal_mode_synthesis_20260826_v2/synthesis.json`
- structural challengers: `reports/structural_challenger_20260827_v1/report-source.md`

Machine-readable family 판정은 `artifacts/promotion_retroaudit_20260827_v1/family_reclassification_ledger.json`, 미충족 요건은 `gap_matrix.json`, 주장-출처 매핑은 `claim_source_ledger.json`에 있다.

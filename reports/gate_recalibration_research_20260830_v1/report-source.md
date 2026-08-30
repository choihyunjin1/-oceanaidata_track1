# 허용치 재보정 및 병렬 구조 실험 보고서

## 결론

기존 승격 게이트에는 공식 점수 변환이나 비용 근거가 없는 고정 허용치와 `모든 구간 개선` 식 결합 조건이 섞여 있었고, 그 결과 **P2의 과거 두 후보를 과도하게 탈락시킨 사실**이 확인됐다. 그러나 새 기준으로 별도 실행한 P1·P2·P3 후보 중 곧바로 제출할 새 챔피언은 없었다.

- 과거 P2 Gaussian copula와 state-conditioned copula는 각각 pooled RMSE 개선 `0.010616065°C`, `0.003459176°C`이고 paired 90% 구간도 전부 개선 방향이다. 둘은 `HIGH_VALUE_CHALLENGER_RESEARCH_ONLY`로 복원한다.
- 새 P1 add-only event precision LCB는 pooled F1이 `-0.002380580`이지만 의존성 보존 구간이 `[-0.017810465, +0.013951394]`로 0을 가른다. **악화 확정이 아니라 불확실**이다.
- 새 P2 availability-aware continuous sparse copula는 RMSE가 `+0.001990430°C` 악화했고 90% 구간 `[+0.000661780, +0.004967253]`도 전부 악화 방향이다. **명확한 연구-only 손실**이다.
- 새 P3 selection-matched sparse Bayesian abstention은 RMSE benefit이 `-0.003475071m`, 90% 구간 `[-0.009929381, +0.003149363]`이다. **악화 확정이 아니라 불확실**이다.

따라서 이번 사이클의 가장 가치 있는 자산은 새 모델이 아니라, 잘못 닫혔던 **P2의 두 잠금 challenger**와 앞으로 적용할 **metric-aligned 승격 규칙**이다. 모든 표면이 이미 연구에 노출된 historical surface이므로 `confirmed`나 `submission-ready champion`으로 부르지 않는다.

## 무엇을 바꿨는가

새 정책은 효능 판단과 유효성 판단을 분리한다.

1. 키·행·순서·정답·비교기 lineage, 누수, finite/domain, 사전등록, 재시도 금지는 hard validity로 유지한다.
2. 효능의 단일 primary는 문제별 공식 형식과 같은 pooled metric으로 둔다.
   - P1: pooled row micro-F1
   - P2: pooled all-row temperature RMSE
   - P3: pooled all-row six-lead Hs RMSE
3. 월·계절·관측소·수심·lead·window는 공식 mixture 또는 실제 안전비용 근거가 없는 한 transport diagnostic으로만 쓴다.
4. `0.003°C`, `0.01m`, `2/3 windows`, `all groups improve` 같은 수치는 공식 점수 변환, 사전 noise study, 사용자 비용, 정식 비열등성 설계가 없으면 새 hard gate로 쓰지 않는다.
5. 결과를 본 뒤 과거 one-shot을 다시 열거나 임계값을 느슨하게 바꾸지 않는다. 과거 산출물은 그대로 두고 해석만 재분류한다.

이 원칙은 scoring target과 평가 규칙을 맞춰야 한다는 Gneiting & Raftery(2007), 선택 기준 자체가 과적합될 수 있다는 Cawley & Talbot(2010), paired loss differential의 의존성을 보존해야 한다는 Diebold & Mariano(1995) 및 Politis & Romano(1994), 비열등성 허용치는 결과 전에 정해야 한다는 Schuirmann(1987)의 설계 원칙에 따른다. Worst-group 목적은 pooled risk와 다른 목적이므로 Sagawa et al.(2020)처럼 그 목적을 실제로 채택할 때만 hard requirement가 된다. 자세한 출처-주장 연결은 `claim-source-ledger.md`에 있다.

## 0-fit 과거 게이트 재생

`gate-replay.json`은 저장된 aggregate만 읽었고 fit, prediction-row/raw-data read, 공식 입력 read, CSV 생성, upload는 모두 0이다.

| 후보 | Primary benefit | 새 판정 | 해석 |
|---|---:|---|---|
| P1 event-balanced SupCon | `-0.164874110 F1` | `PRIMARY_HARM_RESEARCH_ONLY` | pooled와 모든 window/type이 함께 악화해 exact recipe 종료 유지 |
| P2 Gaussian copula conditional mean | `+0.010616065°C` | `HIGH_VALUE_CHALLENGER_RESEARCH_ONLY` | pooled 및 paired CI는 유리, Nov-Dec 회귀는 transport risk |
| P2 state-conditioned copula | `+0.003459176°C` | `HIGH_VALUE_CHALLENGER_RESEARCH_ONLY` | pooled 및 paired CI는 유리, JJA 회귀는 diagnostic |
| P3 CatBoost confirmation | `-0.007974131m` | `PRIMARY_HARM_RESEARCH_ONLY` | benefit CI 전체가 음수 |
| P3 masked SSL | `-0.314155238m` | `PRIMARY_HARM_RESEARCH_ONLY` | benefit CI 전체가 음수 |

재분류는 과거 config/result를 변경하지 않는다. 두 P2 후보를 “확정 우승”으로 승격한 것도 아니다. 제한된 공식 probe를 나중에 승인받아 쓸 때 가장 정보가치가 높은 후보로 복원했을 뿐이다.

## 새 병렬 one-shot 결과

### P1: add-only hierarchical event precision LCB

- Q2→Q3에서는 `+0.008106758 F1`, Q2+Q3→Q4에서는 `-0.017499282 F1`로 방향이 뒤집혔다.
- pooled 287,862행에서 anchor `0.902917024`, candidate `0.900536444`, ΔF1 `-0.002380580`이다.
- 14개 event/470행을 추가해 TP 188, FP 282, precision `0.4`였다. 순수 add-only의 정확한 개선 조건인 `proposal precision > anchor F1/2`의 `0.451458512`를 넘지 못했다.
- event-preserving joint KST-day block bootstrap 5,000회는 0을 가르므로 `INCONCLUSIVE_RESEARCH_ONLY`다.
- fit 2, threshold/HPO/retry 0, anchor removal 0, raw/official/CSV/upload/deletion 0이다.

해석: 임의의 최소 이벤트 수나 모든 분기 개선 조건으로 탈락시킨 것이 아니다. pooled 효과와 불확실성이 실제로 결론을 못 냈다. Q4의 낮은 precision은 event score의 transport가 아직 부족하다는 직접 증거다.

### P2: availability-aware continuous sparse copula

v1은 첫 fold에서 `candidate left the finite physical temperature domain`으로 기술 실패했다. 진단 결과, `45°C`를 넘은 18행은 후보가 새로 만든 값이 아니라 frozen reference에 이미 있던 값이었고 후보는 그 18행에서 exact no-op이었다. 신규·active 위반은 0이었다.

v1은 실패 receipt로 봉인했다. v2는 모델·예측·split·metric·correction bound를 바꾸지 않고, “기존 reference 극값은 exact unchanged일 때만 허용”하도록 guard만 수리한 새 ID다.

- pooled 69,850행: reference `3.085786848°C`, candidate `3.087777279°C`, ΔRMSE `+0.001990430°C`
- 7-day within-window KST-day block bootstrap 90% 구간 `[+0.000661780, +0.004967253]°C`, 개선확률 `0.0028`
- outer fits 3, edge estimates 21, inner/HPO 0/0
- correction 최대 `0.2°C`; official/query/CSV/upload/hard deletion 0
- independent QA `28/28 PASS`

해석: guard 수리는 정당했지만 과학적 후보는 pooled primary에서 명확히 악화했다. 이 exact recipe는 닫는다.

### P3: selection-matched sparse Bayesian abstention

- 157 cases × 6 leads = 942행
- incumbent `0.811219163m`, candidate `0.814694233m`, benefit `-0.003475071m`
- window-stratified contiguous anchor-day block 90% 구간 `[-0.009929381, +0.003149363]m`
- active correction 495/942, exact incumbent 447/942, candidate fits 3, HPO 0
- 2025-H1만 `+0.005443399m` benefit이었고 storm/winter는 각각 `-0.004290281m`, `-0.010027203m`였다. 이들은 veto가 아니라 transport 진단이다.
- official/context/sample/baseline/score/submission/hidden read 0, CSV/upload/deletion 0, independent QA `14/14 PASS`

해석: 임의의 `0.01m`, `2/3 windows`, worst-lead cap을 없애도 CI가 0을 가르므로 불확실이다. 2025-H1의 조건부 이득이 어떤 정보 축에서 발생했는지 분리하지 않은 채 threshold만 다시 만지는 것은 가치가 낮다.

## 이상치에 대한 결론

이상치 제거는 이번 데이터에서 기본값으로 채택하지 않는다.

- P2의 18개 극값은 후보가 만든 이상치가 아니라 기존 comparator의 exact no-op 행이었다. 이를 일괄 삭제하면 평가 모집단과 comparator가 바뀐다.
- P3의 jump-return flag 2개도 센서 오류의 정답표가 아니다. cohort·weight·삭제에 사용하지 않았다.
- P1도 positive event나 raw anomaly signal을 삭제하지 않았다.

앞으로도 “물리적으로 드문 값”과 “검증된 센서 오류”를 구분한다. 별도 센서 오류 규칙을 사전등록하고 독립 표면에서 검증하기 전에는 hard delete 대신 flag, robust loss, abstention의 진단 축으로만 쓴다.

## 다음 연구 판단

1. **P2:** 가장 높은 정보가치는 새 exact recipe가 아니라 복원된 Gaussian copula와 state-conditioned copula다. 다음 공식 기회를 쓸 경우 둘 중 하나를 고정 probe로 삼고, 결과에 따라 다음 행동이 달라지는 비교 설계를 먼저 적는다.
2. **P1:** add-only score가 Q3에서 성공하고 Q4에서 무너진 원인을 event transport feature 또는 새로운 미노출 labeled window로 검증한다. 같은 exposed Q2/Q3/Q4에서 threshold를 더 찾지 않는다.
3. **P3:** 2025-H1 이득과 storm/winter 손실을 설명하는 새로운 정보 축이 없으면 이 family를 더 미세 조정하지 않는다. lead/window hard veto를 되살리는 것도 해법이 아니다.
4. **공통:** official score와 raw metric의 변환·표시 해상도가 확보되기 전에는 “3점”을 `3°C`나 `0.03m` 같은 raw-unit gate로 바꾸지 않는다.

이번 사이클은 공식 제출·CSV 생성·업로드를 수행하지 않았다. 코드·설정·결과·QA는 아직 커밋하거나 push하지 않았다.

## 재현성과 봉인 한계

- P1은 exclusive attempt lock과 config/module/runner hash가 있으나 result를 별도 외부 immutable seal로 다시 묶지는 않았다.
- P2 v2는 `attempts=1`, retry/tuning=false인 execution receipt와 28/28 독립 QA가 있지만 pre-fit attempt lock과 sealed payload가 없다. 따라서 과학적 결과는 채택하되 “암호학적으로 완전 증명된 one-shot”이라고 과장하지 않는다.
- P3는 attempt lock, config/code hash, payload seal, 14/14 독립 QA가 모두 일치한다.
- gate replay는 현재 입력·policy·runner hash가 맞지만 exclusive output seal은 아니다. 통합 QA가 다섯 primary 값과 부호 변환을 원 aggregate에서 다시 계산했다.

접근량도 주체별로 구분한다. 통합 QA와 gate replay는 aggregate/code만 읽어 raw와 official이 0이다. P1 실행은 등록 OOF를 재사용해 raw 0이었다. P2는 789,408행 `observations.csv`를 v1 실행, guard 진단, v2 실행에서 각각 한 번 열었고, P3는 `train_wave.csv` 118,152행과 `train_atmos.csv` 130,896행을 읽었다. 이들은 허용된 training source이며 공식 test/sample/submission 접근과는 다르다. 모든 실행의 공식 입력 read, prediction CSV output, upload, hard deletion은 0이다.

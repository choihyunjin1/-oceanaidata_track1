# 2026-08-29 P1/P2/P3 병렬 최적화 사이클 최종 보고서

대상: 해양 AI 해커톤 모델 연구·재현성 검토

작성일: 2026-08-29

기준 커밋: `8e0dc9ab22a713596425375985a6fe04f878e325`

## 결론

이번 사이클에서는 공식 제출로 승격할 모델이 나오지 않았다. 다만 세 실험의 결론은
서로 다르다.

- **P1 — `NO_GO_PRECONFIRM`**: 32개 Sobol 후보와 top-2 추가 seed를 포함한 36 fits를
  정상 완료했다. 선택 후보는 모든 월에서 양의 ΔF1과 anchor 제거 0을 달성했지만 pooled
  ΔF1이 `+0.000565637`로 고정 승격선 `+0.003`에 미달하여 Q3/Q4 전에 중단했다.
- **P2 — `NO_GO_CLOSE_FAMILY`**: `243 × 3` 후보 평가와 84 PLS fits를 정상 완료했다.
  pooled ΔRMSE는 `-0.002041992 °C`였지만 3개 outer fold 중 개선 1, 회귀 1, 무변화 1이고
  한 inner selection이 부적격이어서 close-family로 닫았다.
- **P3 — `INVALID / TERMINAL TECHNICAL FAILURE`**: 74 fits 뒤 75번째 시도에서 사전
  등록 grid의 `Ordered + Depthwise` 비호환 조합으로 종료됐다. 이는 성능 NO_GO가 아니며
  어떤 후보 순위나 gate 결론도 만들지 못했다.

따라서 현재 champion 또는 제출물을 교체할 과학적 근거는 없다. P1과 P2는 유효한 음성
결과로 보존하고, P3는 새 실험 ID에서 **모든 후보를 실제 one-tree fit으로 검증한 뒤**
valid-combination grid만 실행해야 한다.

## 범위와 경계

이 보고서는 세 historical/local 실험의 코드, 고정 config, 작은 결과 receipt, 해시와
독립 QA만 다룬다. 공식 hidden/test/sample/submission 값은 읽지 않았고 제출 CSV를
만들거나 업로드하지 않았다. 대용량 NPZ/prediction/checkpoint/raw data와 attempt lock은
Git 보존 범위에서 제외한다.

방법론 판단은 2026-08-29에 확인한 1차 문헌·공식 문서를 사용했다. 성능 수치는 외부
문헌의 결과가 아니라 저장소의 봉인된 local artifact에서 나온 값이다.

## P1: MS-TCN 32-point Sobol HPO

### 설계

`random_base2(m=5)`로 32개 저불일치 Sobol 점을 사전 봉인하고 Q2에서 각 후보를 한 seed,
150 epochs로 full-fidelity 평가했다. SciPy 문서는 `random_base2`가 `2^m`개 점을 생성해
Sobol balance property를 보존한다고 명시한다. Bergstra와 Bengio의 연구는 일부
hyperparameter만 실제 성능을 좌우할 때 고정 격자보다 무작위 탐색이 계산 예산을 더
효율적으로 사용할 수 있음을 보였다. 이 근거는 탐색 설계를 정당화하지만 본 데이터의
성능 향상을 보장하지는 않는다.

Discovery 32 fits 뒤 사전 규칙에 따라 top-2 후보에 seed 두 개씩을 추가했다. 선택과
확인을 분리하기 위해 고정 pre-confirm gate를 적용했고, gate를 넘을 때만 Q3/Q4를
실행하도록 했다. 이는 유한 validation criterion을 반복 최적화할 때 selection bias가
발생할 수 있다는 Cawley와 Talbot의 경고에 대응한다.

### 실측 결과

- 총 36 fits: discovery 32 + top-2 추가 seed 4
- 실행 시간: `26,183.483 s` (`7.273 h`), RTX 5090
- 선택: `trial_18`, epoch 150, threshold 0.8, seeds
  `20260827 / 20260839 / 20260863`
- Q2 control F1: `0.867675736`
- Q2 candidate F1: `0.868241373`
- pooled ΔF1: `+0.000565637`
- 월별 ΔF1: April `+0.001005635`, May `+0.000294857`, June `+0.001336783`
- frozen-anchor positive removal: `0`

월별 양수와 removal 0은 통과했지만 pooled ΔF1 `>= +0.003`을 통과하지 못했다. 따라서
`STOP_BEFORE_CONFIRMATION`; Q3/Q4 fit과 artifact는 0이다. 독립 QA 28/28, focused tests
36개와 Ruff가 통과했다.

해시 핵심: config `c9ff0cee…14316b`, runner `155e7719…817401`, design
`98a52582…e723ed`, aggregate `fb42fe28…4fcf98`, QA `437ee74d…8cc46`.

## P2: nested PLS capacity grid

### 설계

세 historical outer window 각각에 대해 243개 고정 후보를 평가했다. 총 candidate
evaluation은 729, rotation-point evaluation은 2,187, 실제 PLS fit은 inner 81 + outer
3 = 84였다. outer label 전에 prediction을 봉인하고, inner selection과 outer evaluation을
분리했다. 이러한 분리는 model-selection criterion 자체의 과적합과 후속 성능평가
selection bias를 구분해야 한다는 Cawley와 Talbot의 1차 근거와 일치한다.

### 실측 결과

- reference RMSE: `3.085786848 °C`
- candidate RMSE: `3.083744857 °C`
- pooled ΔRMSE: `-0.002041992 °C`
- fold ΔRMSE: Sep–Oct `-0.003483191`, Jul–Aug `+0.000209940`, Nov–Dec `0`
- layer ΔRMSE: L2 `-0.008162196`, L3 `-0.005793903`, L4 `-0.000321594`
- 5,000 KST-day bootstrap q95 upper: `-0.001051630 °C`
- 실행 시간: `1,902.745 s` (`31 m 42.745 s`)

pooled, all-layer, bootstrap, cosine, correction-size gate는 통과했다. 그러나 Sep–Oct
inner Δ가 정확히 0이라 `all_inner_selections_eligible=false`; fold 방향은 개선 1,
회귀 1, 무변화 1이라 `two_of_three_folds=false`; Jul–Aug의 양의 ΔRMSE 때문에
`worst_fold=false`였다. 결론은 `NO_GO_CLOSE_FAMILY`다. 독립 QA와 6 focused tests,
Ruff가 통과했다.

해시 핵심: config `f5fff66a…0ba63`, source `70e5b3f9…27b5a`, runner
`466c76ea…5be3`, result `d08c79d5…76296`, commitment `e9774c29…7b633`, QA
`680569fe…0992`.

## P3: CatBoost ordered HPO v1

### 설계와 실패

사전 grid는 48 challengers를 300/900/2500-tree successive-halving으로 줄이고,
selection·confirmation·gate-pass refit을 합쳐 최대 174 fits로 제한했다. Successive
Halving은 유망 후보에 더 많은 자원을 배분하는 hyperparameter allocation 방법이라는
Jamieson과 Talwalkar의 1차 근거를 따랐다. CatBoost 논문은 ordered boosting이 prediction
shift와 특정 target-leakage 문제를 줄이기 위한 permutation 기반 방법이라고 설명한다.

그러나 parameter object 구성만 수행한 static preflight는 fit-time 조합 제약을 검증하지
못했다. CatBoost v1.2.10 소스는 grow policy가 `SymmetricTree`가 아니면 boosting type이
`Plain`이어야 한다고 강제하며, 그렇지 않으면 정확히 `Ordered boosting is not supported
for nonsymmetric trees.`를 발생시킨다. 고정 grid에는 이 제약을 위반하는 12개
`Ordered + Depthwise` 후보가 들어 있었다.

- 성공 fits: `74`
- 실패 시도: `75th`, `challenger_37`, 첫 selection fold
- 마지막 진행 receipt: 70/98 rung-300 fits at `6,878.5 s`
- rung 완료 / ranking / selection metric / confirmation: 모두 `0`
- 판정: `INVALID / TERMINAL TECHNICAL FAILURE`

실행 config/grid/lock/artifact는 변경하지 않았고 재실행도 금지했다. 사후에는 runner가
이 조합을 model fit 전 fail-fast하도록 고쳤으며 dedicated 7 tests와 Ruff가 통과했다.
이는 v1 결과를 복구하거나 과학적 결론으로 바꾸지 않는다.

해시 핵심: grid `98710828…571e`, resource amendment `70742c1a…b079`, 실행 config
`d162ece2…0adc`, 실행 runner `b275f0a2…e6f0`, attempt lock `51327c0d…aec7`, 사후
runner `f7b8e2c3…3802`.

## 공통 QA와 해석

세 실험 모두 official/hidden/test/sample/submission 값 접근은 0행, CSV 생성과 upload는
0이다. P1/P2는 사전 gate가 약한 local 개선을 confirmation 또는 제출로 자동 승격시키는
것을 막았다. P3는 model-space 평가 이전의 계약 오류이므로 성능 결과로 비교해서는 안
된다.

이번 사이클의 핵심 교훈은 “탐색량을 늘리면 충분하다”가 아니다. 탐색 전 조합
실행가능성, 선택과 확인의 분리, fold 방향성, effect size를 함께 봉인해야 한다. 특히
P1의 모든 월 양수는 후속 가설 신호로 보존할 가치가 있지만, pooled effect가 승격선의
약 18.9%에 불과해 현재 후보를 champion으로 채택할 근거는 아니다.

## 다음 단계

1. P3는 새 ID `p3_catboost_valid_hpo_20260829_v2`에서 `Plain+SymmetricTree`,
   `Plain+Depthwise`, `Ordered+SymmetricTree`만 허용하고 모든 후보에 deterministic
   one-tree smoke fit을 완료한 후 historical lock을 생성한다.
2. P1은 현재 Sobol 공간을 추가 튜닝하지 않는다. 별도 one-shot에서만 사전 정의
   `station × layer × time-regime` 강건 손실을 검토하고 Q2/Q3/Q4 분리를 유지한다.
3. P2 close-family는 닫는다. 다음 가설은 기존 선형 구조와 독립적인 0-fit support audit을
   먼저 통과해야 하며, 공식 입력 접근 경계가 필요한 경우 별도 승인 전에는 실행하지
   않는다.

## 출처

- James Bergstra & Yoshua Bengio, “Random Search for Hyper-Parameter Optimization,”
  JMLR 13, 2012: https://www.jmlr.org/papers/v13/bergstra12a.html
- SciPy community, `scipy.stats.qmc.Sobol.random_base2` documentation, accessed
  2026-08-29: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.random_base2.html
- Gavin C. Cawley & Nicola L. C. Talbot, “On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation,” JMLR 11, 2010:
  https://www.jmlr.org/papers/v11/cawley10a.html
- Liudmila Prokhorenkova et al., “CatBoost: unbiased boosting with categorical
  features,” NeurIPS 2018:
  https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- CatBoost, “Parameter tuning,” accessed 2026-08-29:
  https://catboost.ai/docs/en/concepts/parameter-tuning
- CatBoost v1.2.10 source, `catboost_options.cpp`, accessed 2026-08-29:
  https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/private/libs/options/catboost_options.cpp
- Kevin Jamieson & Ameet Talwalkar, “Non-stochastic Best Arm Identification and
  Hyperparameter Optimization,” AISTATS/PMLR 51, 2016:
  https://proceedings.mlr.press/v51/jamieson16.html

검색 중단 기준: 세 실험의 설계 정당화, P3 실패 원인, selection-bias 경계에 대해 1차
근거가 확보되었고 추가 문헌이 terminal 판단이나 다음 실행 순서를 바꿀 가능성이 낮아
중단했다.

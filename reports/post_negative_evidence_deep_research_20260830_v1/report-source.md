# 실패 증거 이후의 다음 실험 딥리서치

기준 커밋: `54a0a7ab98b58abc13d6b7c99b84a9f2089259a6`

## 결론

다음 사이클의 우선순위는 **P3 확인 파이프라인 수리 → P2 비선형 copula 조건부 평균 → P1 event-balanced supervised contrastive + F1 정렬 head**다. 세 방향 모두 기존 실패 원장에서 닫힌 정확한 축을 반복하지 않는다.

1. **P3가 가장 값싸고 증거가 강하다.** 138-fit 선택을 다시 할 이유가 없다. `challenger_21`은 선택 구간에서 control보다 RMSE가 0.0228625 낮았고, 실험은 모델 열세가 아니라 확인 입력 schema의 `current_hs`/`single_prediction` 누락으로 끝났다. 후보·파라미터를 동결하고 확인 contract만 수리한다.
2. **P2는 작은 CPU 실험 하나만 허용한다.** 선형 PLS/OAS/GP/bridge 반복이 아니라, rank-Gaussian marginal transform과 조건부 분포를 이용한 원척도 조건부 평균을 검증한다. train-only support audit은 실행 가능성만 입증했으며 성능을 입증하지 않았다.
3. **P1은 고위험 연구 화면(screen)이다.** 일반 BCE, 전역 threshold, Group-DRO, synthetic corruption을 반복하지 않는다. 실제 anomaly event 단위 균형표집과 supervised contrastive representation을 쓰고, 비분해 가능한 F1과 학습 목적의 불일치를 줄이는 calibrated F1 head를 결합한다.

이 문서는 다음 실험을 **승인하거나 실행하지 않는다**. 공식 test/sample/submission/hidden label을 읽지 않았고 CSV 생성·업로드도 0건이다.

## 현재 증거의 해석

| 문제 | 현재 가장 중요한 증거 | 올바른 분류 | 다음 판단 |
|---|---|---|---|
| P1 | fixed Group-DRO MS-TCN이 Q2 3-seed에서 pooled ΔF1 `-0.0134805`, 최소 월별 `-0.0235490`, station share `0.802817` | 정확한 objective/decoder 조합의 `NO_GO_Q2` | representation과 metric 정렬을 동시에 바꾸는 저비용 screen만 허용 |
| P2 | 47,216 complete timestamps, 최소 seasonal-layer-coordinate support 2,353, duplicate/nonfinite 0 | `TRAIN_ONLY_SUPPORT_PASS`; 성능 증거 아님 | 비선형 copula conditional mean을 chronological outer folds에서 검증 |
| P3 | `challenger_21` 선택 RMSE `0.5964341` vs control `0.6192966`; 첫 confirmation 뒤 schema KeyError | `INVALID_TERMINAL_TECHNICAL_FAILURE`; 과학적 NO_GO 아님 | 탐색 재실행 없이 contract test 후 고정 후보 확인만 수행 |

전체 세부 실패 분류는 `../negative_evidence_registry_20260830_v1/report-source.md`와 `failure-ledger.json`을 정본으로 삼는다. “정확한 조합의 실패”를 “모델 계열 전체의 불가능”으로 확대하지 않는다.

## 우선순위 1 — P3 confirmation contract repair

### 연구 질문

이미 선택된 `challenger_21`의 개선이 사전등록된 세 historical confirmation window에서도 재현되는가?

### 고정할 것

- 후보: `challenger_21`
- CatBoost: Plain boosting, Depthwise grow policy, depth 9, learning rate 0.02, L2 20, RSM 0.75, Bayesian bagging temperature 0.2
- control, feature set, chronological windows, postprocess, gate
- 138-candidate search 결과와 hash

### 고칠 것

새 실험 ID를 `p3_catboost_confirmation_contract_repair_20260830_v3`로 분리한다. attempt lock 전에 synthetic fixture와 canonical historical schema를 confirmation engine 끝까지 흘려 다음을 assert한다.

- 정확한 컬럼 순서: pair keys, frozen router columns, `current_hs`, `single_prediction`
- dtype/finite, pair-key uniqueness, 행 수 보존
- selection과 confirmation projection의 동일 schema hash
- missing column이면 **모델 fit 전에** fail-fast

contract가 통과한 뒤 control과 frozen challenger를 세 confirmation window에만 실행한다. 선택 탐색과 파라미터 조정은 재실행하지 않는다. 최대 6 fits(control/challenger × 3 windows)로 제한한다.

### 승격/중단 기준

- 세 confirmation window 모두 challenger RMSE가 control보다 낮음
- pooled ΔRMSE < 0
- 기존 slice/lead/stability guard 전부 통과
- 결과, schema, config, prediction lineage hash 완전 일치
- 하나라도 실패하면 해당 고정 후보를 `NO_GO_CONFIRMATION`으로 닫고 결과 기반 재튜닝 금지

CatBoost 공식 문서는 validation set으로 best iteration을 선택하고 grow policy·depth·bagging 관련 파라미터를 명시적으로 관리할 것을 권한다. 여기서는 이미 선택이 끝났으므로 그 문서를 새 탐색의 근거가 아니라 **고정 후보 재현성**의 근거로 사용한다. ML Test Score와 TFX Data Validation 연구는 schema와 pipeline contract 오류가 모델 품질과 별개의 운영 실패축임을 뒷받침한다.

## 우선순위 2 — P2 nonparanormal copula conditional mean

### 연구 질문

같은 시각의 관측 가능 층과 incumbent 예측을 조건으로 할 때, residual의 비선형 rank-Gaussian 의존구조가 기존 선형 조건부 보정보다 안정적으로 RMSE를 낮추는가?

### 설계

새 실험 ID는 `p2_gaussian_copula_conditional_mean_20260830_v1`이다.

1. incumbent/OAS/rank1 예측은 동결하고 historical residual만 학습한다.
2. train fold 안에서만 계절 또는 월별 empirical marginal transform을 적합한다.
3. Kendall-tau 기반 latent Gaussian covariance를 추정하고 작은 고정 shrinkage set만 비교한다.
4. 관측 가능한 같은 시각의 temperature/salinity/depth와 incumbent prediction을 조건으로 missing layer residual의 조건부 분포를 구한다.
5. inverse marginal이 비선형이므로 `inverse(E[z|x])`를 원척도 평균으로 간주하지 않는다. 고정 Gauss-Hermite quadrature로 `E[residual|x]`를 원척도에서 근사한다.
6. 사전등록 3개 chronological outer fold만 사용한다. query/test support나 공식 입력은 읽지 않는다.

### 승격/중단 기준

- transform이 finite·monotone이고 covariance가 PSD이며 condition-number guard 통과
- pooled ΔRMSE < 0, 3개 fold 중 최소 2개 개선
- 어떤 target layer도 control보다 0.001°C를 초과해 악화하지 않음
- outer-fold paired bootstrap 상한이 0 미만이거나, 표본이 부족하면 더 보수적인 기존 gate 적용
- wide HPO 금지: 고정된 소수 shrinkage와 marginal granularity만 비교하고 한 사이클에서 종료

Gaussian copula regression과 nonparanormal 연구는 알려지지 않은 단조 marginal transform 아래의 의존구조·예측을 다룬다. 그러나 대회 데이터로의 성능 전이는 문헌 사실이 아니라 **이 실험이 검증해야 할 추론**이다.

## 우선순위 3 — P1 event-balanced SupCon + F1-aligned head

### 연구 질문

행 단위 BCE가 지배적인 normal class와 station imbalance에 끌리는 문제를, 실제 anomaly event 단위 표집과 supervised contrastive representation으로 줄일 수 있는가?

### 설계

새 실험 ID는 `p1_event_balanced_supcon_f1_head_20260830_v1`이다.

- 실제 라벨의 연속 anomaly event를 sampling unit으로 사용하고 synthetic anomaly 생성은 금지한다.
- anomaly type 5종을 supervised positive class로 삼고, station×layer×season을 맞춘 hard-normal window를 구성한다.
- event가 fold 사이에 겹치지 않도록 event-disjoint chronological validation을 만든다.
- 작은 temporal encoder를 supervised contrastive loss로 학습한 뒤 proposal head를 붙인다.
- proposal head는 ordinary BCE + global threshold가 아니라 calibrated linear-fractional/F1 surrogate 또는 conditional-probability ranking 기반 expected-F1 top-k를 사용한다.
- 기존 anchor-union decoder는 동결하고 anchor removal을 0으로 유지한다.

### 단계별 예산과 gate

첫 단계는 1 seed, 20–30 epochs의 화면만 허용한다.

- all-window ΔF1 > 0
- anomaly-type macro-F1 개선 및 embedding separation 악화 없음
- 추가 proposal precision이 각 window incumbent F1의 절반보다 큼
- anchor removal 0
- changed-row station share ≤ 0.8

이 화면을 모두 통과할 때만 별도 승인 사이클에서 3-seed 확인을 고려한다. Q2/Q3/Q4가 반복 사용된 상태이므로 통과하더라도 결과는 `RESEARCH_ONLY`로 표시하고 공식 일반화 증거로 과장하지 않는다.

Supervised contrastive time-series 연구는 제한된 라벨과 high-frequency series에서 augmentation·class spread를 활용하는 방법을 제안한다. F1 최적화 연구는 F-measure가 비분해 가능하며 conditional-probability ranking 또는 calibrated surrogate가 직접적인 대안이 될 수 있음을 보인다. 해양 anomaly 데이터에서의 성공은 아직 입증되지 않았다.

## 다시 실행하지 않을 것

- P1: ordinary BCE + global threshold, threshold/veto/bridge/density 규칙, synthetic corruption, fixed Group-DRO objective, 동일 32-point Sobol space
- P2: 기존 PLS/OAS/GP/bridge grid, linear rank1/heave recipe, query boundary를 요구하는 two-sided bridge
- P3: 138-candidate CatBoost selection, future-wind/KMA alpha slicing, frozen-KMA MOS, 현 ERA5 solution-gate recipe
- 공통: 결과를 본 뒤 탐색공간·fold·gate를 바꾸는 재튜닝, random CV로 chronological validation 대체, 공식 점수를 로컬 gate의 사후 조정값으로 사용

## 실행 순서와 의사결정

1. **P3 contract-only preflight**: 수초~수분, 0 fit. 실패 시 즉시 종료.
2. **P3 frozen confirmation**: 최대 6 fits. 통과하면 가장 먼저 승격 후보가 된다.
3. **P2 copula CPU pilot**: 고정 3 outer folds와 작은 shrinkage set. close-family 성격 때문에 한 번만 실행한다.
4. **P1 low-fidelity GPU screen**: P3와 P2 결과를 기다리지 않아도 되지만, 본격 3-seed 확인은 screen 전 gate 통과 뒤 별도 결정한다.

시간 추정은 현재 로컬 실행 이력에 근거한 계획값이며 보장값이 아니다. 정확한 wall time은 preflight에서 데이터 행 수와 feature matrix 크기만 확인해 갱신한다.

## 1차 출처

- [Prokhorenkova et al., CatBoost: unbiased boosting with categorical features, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)
- [CatBoost official parameter tuning documentation](https://catboost.ai/docs/en/concepts/parameter-tuning)
- [Breck et al., The ML Test Score, IEEE Big Data 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
- [Baylor et al., Data Validation for Machine Learning, SysML 2019](https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf)
- [Sculley et al., Hidden Technical Debt in Machine Learning Systems, NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [Cawley and Talbot, On Over-fitting in Model Selection, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Dey and Zipunnikov, High-Dimensional Gaussian Copula Regression, Statistica Sinica 2018](https://www3.stat.sinica.edu.tw/sstest/j28n2/j28n219/j28n219.html)
- [Liu, Lafferty and Wasserman, The Nonparanormal, JMLR 2009](https://www.jmlr.org/papers/v10/liu09a.html)
- [Ghosal et al., Gaussian copula function-on-scalar regression in RKHS, Journal of Multivariate Analysis 2023](https://doi.org/10.1016/j.jmva.2023.105226)
- [Noroozizadeh et al., Temporal Supervised Contrastive Learning, PMLR 2023](https://proceedings.mlr.press/v225/noroozizadeh23a.html)
- [Deldari et al., Supervised Contrastive Few-Shot Learning for High-Frequency Time Series, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25863)
- [Khosla et al., Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- [Bao and Sugiyama, Calibrated Surrogate Maximization of Linear-fractional Utility, AISTATS 2020](https://proceedings.mlr.press/v108/bao20a.html)
- [Natarajan et al., Optimal Classification with Multivariate Losses, ICML 2016](https://proceedings.mlr.press/v48/natarajan16.html)
- [Bergmeir, Hyndman and Koo, A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction, CSDA 2018](https://robjhyndman.com/publications/cv-time-series/)

# 해양 해커톤 모델 돌파구 딥리서치

**팀:** 분당독고다이  
**작성일:** 2026-08-26  
**범위:** P1 이상탐지, P2 수온복원, 검증·제출전략  
**보호 범위:** P3 ERA5 고정 실험은 변경·재실행·데이터 열람 없이 제외. 공식 test/sample/submission/candidate 파일도 열람·생성·업로드하지 않음.

> [결론] 지금의 병목은 모델 용량이 아니라 **비교기·후처리·목적함수·검증면의 불일치**다. 새 학습 전에 P2의 outer-exposed/adaptively tuned comparator를 분리하고 P1의 inner/outer/deployment anchor 후처리를 통일해야 한다. 그 다음 P1은 행 단위 F1의 한계효용을 학습하는 비파괴 구간 구조로 간다. P2의 shrinkage multivariate FPCA는 layer ordinal이 아니라 실제 nominal depth로 fold별 격자를 구성할 수 있고 target cross-covariance가 식별될 때만 진행한다. 두 가설 모두 0-fit 상한검사 후 사전등록된 두 사이클 안에서만 검증한다.

## 1. 의사결정 요약

### 지금 할 것

- **0-fit 검증교정 — 최우선:** P2는 사후 최적화 exact comparator와 실제 frozen deployed incumbent를 분리하고 architecture-matched base를 함께 둔다. P1은 모든 historical outer를 deployment postprocess로 다시 decode해 anchor semantics를 통일한다.
- **P1 1순위 — F1-aware change-point proposal rescue:** frozen incumbent를 보존하고 PELT-L1, NOT/선형 변화점으로 후보 구간을 만든다. proposal의 단순 겹침 여부가 아니라 추가 TP·FP와 이벤트 커버리지를 학습하고, 겹치지 않는 조합의 누적 예상 TP·FP로 exact F1을 계산해 가장 좋은 집합만 더한다.
- **P2 1순위 — 조건부 shrinkage hydrographic FPCA:** layer 번호를 연도 간 고정 좌표로 취급하지 않는다. outer별 nominal-depth regime에서 학습·배포에 공통으로 관측되는 T/S와 숨은 온도층 2·3·4의 공분산이 식별될 때만 조건부 복원을 시도한다. 기존 깊이 선형보간은 안전한 anchor로 남기고 blend는 nested OOF에서 한 번만 잠근다.
- **검증 1순위 — transport gate:** P1은 행-pooled F1을, P2는 공식과 같은 row-pooled RMSE를 주지표로 둔다. station/layer/day 균형 지표는 안전장치이지 선택 목적함수가 아니다.

### 지금 하지 않을 것

- P1의 현재 72-fit segment-rescore 계획을 목적함수 수정 없이 실행하지 않는다.
- 이미 실패한 generic TCN, hard typed semi-Markov, density-ratio correction, 대형 HPO를 반복하지 않는다.
- P2의 naive temperature-only EOF, 일반 선형 tide/RTS, hard monotonic projection, 대형 Transformer를 재시도하지 않는다.
- 남은 공개 제출을 threshold·blend 탐색에 쓰지 않는다. 로컬 gate를 통과한 champion과 구조적으로 독립인 hedge만 제출한다.

### 판단 상태

| 항목 | 판단 | 근거 수준 |
|---|---|---|
| P2 현 exact comparator로 family 우월성 판정 | NO-GO | 동일 OOF 사후최적화가 직접 확인됨 |
| P1 fold별 postprocess가 다른 현 비교 | NO-GO | inner/outer/deploy 설정 불일치가 직접 확인됨 |
| P1 현재 v1 proposal-rescore 그대로 실행 | NO-GO | 코드 목적함수 불일치가 직접 확인됨 |
| P1 수정형 F1-aware proposal rescue 연구 | GO | 로컬 실패구조 + 변화점·F1 추론 이론이 일치 |
| P2 conditional multivariate FPCA 연구 | CONDITIONAL GO | 실제 수심 기반 fold별 격자와 target 공분산 식별성 확인이 선행되어야 함 |
| P2 계절 prior + causal dynamics | CONDITIONAL GO | FPCA Cycle 1 통과 시만 추가 |
| P3 | HOLD / UNTOUCHED | 별도 고정 실험의 계약 보호 |

## 2. 점수 상황과 현실적인 목표

현재 보관된 리더보드 스냅샷에서 분당독고다이는 78.092863점, 3위이며 1위 82.678604점과 4.585741점 차이다. 이 값은 2026-08-26 스냅샷이며 실시간 재확인값이 아니다.

| 문제 | 현재 관측 지표 | 문제별 최고에서 역산한 지표 | 점수 headroom | 해석 |
|---|---:|---:|---:|---|
| P1 | F1 0.793710 | 약 0.907907 | 3.035360점 | 가장 큰 개선원; 장기 offset/drift recall이 핵심 |
| P2 | RMSE 0.541085°C | 약 0.396196°C | 1.817999점 | 수직 구조와 계절 수송성이 핵심 |
| P3 | RMSE 0.607071m | 약 0.583483m | 0.374402점 | 현재 연구 범위 밖; 고정 실험 유지 |

P1의 제출 후보 최소효과 `ΔF1 >= +0.0255`는 계획 환산 약 +0.678점이다. P2의 의미 있는 최소효과 `ΔRMSE <= -0.060°C`는 약 +0.753점, 선호효과 `-0.0759°C`는 약 +0.952점이다. 따라서 P1 최소 + P2 최소는 약 +1.43점, P1 최소 + P2 선호는 약 +1.63점이다. 2~3점 개선에는 이 gate를 넘는 stretch 성과나 P3의 별도 개선까지 필요하다. 이 수치는 목표·의사결정용 환산이며 달성 예측이 아니다.

## 3. 왜 지금까지의 개선이 공식 점수로 운송되지 않았는가

### 3.1 로컬 점수는 절대 보정식이 아니다

- P1 Round-B의 보관된 로컬 OOF는 F1 0.864670, precision 0.951804, recall 0.792152지만 사용자 제공 공식 F1은 0.793710이다. 두 평가는 표본면이 다르므로 차이 0.070960을 단순 교정상수로 쓸 수 없다.
- P2 incumbent의 3개 historical fold `fold_equal_layer_equal_RMSE_C`는 0.749530°C, 사용자 제공 공식 row-pooled original은 0.541085°C였다. 집계·표본이 모두 달라 두 값을 직접 교정할 수 없다. 후보 A/B는 공식에서 각각 0.713520°C, 0.599921°C로 악화했다.
- P2에서는 exact/full-prefix 계보는 두 후보의 실패 방향과 일치했지만 pooled surrogate는 반대 방향을 냈다. 즉 작은 surrogate의 평균 개선보다 **같은 계보·full budget·시간분리**가 더 신뢰할 만했다.

### 3.2 결론

로컬은 후보 생성에는 유효하지만, 다음 네 조건을 함께 만족할 때만 승격 근거다.

1. 공식 지표와 동일한 primary metric.
2. 고정된 historical time blocks와 exact/full-prefix 계보.
3. paired day bootstrap의 방향과 하한/상한 gate.
4. layer/station/event type의 안전장치와 prefix-to-full 부호 일치.

다만 P1 Q2/Q3/Q4와 P2 세 historical block은 이전 연구 가설에 반복 노출된 **hypothesis-exposed surfaces**다. 새 모델 fit에서 target을 제외하고 prediction freeze 뒤 한 번 평가하는 model-fit-held-out 규율은 지키되, bootstrap 구간은 fresh-holdout 추론 CI가 아닌 stability interval로만 해석한다. 진짜 transport 확인은 새 미래 시간블록 또는 사전등록 champion의 공식 1회 제출이다.

공개 리더보드를 반복 질의해 threshold나 blend를 맞추면 holdout에 적응 과적합할 수 있다는 연구 결과도 이 규율을 지지한다.

## 4. P1 코드 감사: 현재 접근이 놓친 것

대상: `src/p1_qc/long_event_segment_proposal_rescore.py`

### 4.1 핵심 P0 과학적 취약점 — proposal target이 공식 F1과 다르다

현재 label 함수는 proposal 내부의 80% 이상이 하나의 eligible event에 속하면 positive로 둔다(40~41행, 675~719행). 이 정의는 **proposal이 실제 이벤트를 얼마나 복구하는지**를 요구하지 않는다. 긴 이벤트의 작은 일부만 덮는 구간도 positive가 될 수 있고, 행 단위 false positive 비용도 target에 직접 들어가지 않는다.

threshold도 proposal-level precision으로 선택된다(약 831~850행). 공식 목적은 row-pooled F1이므로, 학습 target·threshold·공식 metric이 서로 다른 최적점을 가질 수 있다. 더 많은 fit으로 해결될 문제가 아니다.

### 4.2 P1 추가 취약점

- **anchor semantics 불일치:** Round-B outer의 Q2/Q3/Q4 후처리 규칙이 서로 다르고 deployment는 또 다른 `0.2/0.1 + gap0 + minrun12` 규칙이다. inner shelf는 deployment 규칙을 쓰지만 outer 평가는 fold별 frozen p100 prediction을 그대로 쓴다. proposal의 `anchor_positive_fraction`과 connected decoder가 anchor state에 직접 의존하므로 selection·outer·deployment가 서로 다른 문제를 푼다.
- **비교 성숙도 불일치:** challenger는 모든 Q2/Q3/Q4 outer에서 2024년 6월·9월·12월의 같은 세 calibration shelf만 학습한다. incumbent는 fold별 expanding p100이다. 구조와 threshold를 먼저 동결한 뒤, outer마다 그 시점 이전의 cross-fitted proposal labels로 challenger도 expanding refit해야 family 비교가 공정하다.
- `exact_gap_safe_segment_ids`는 입력 순서를 신뢰하며 정렬·단조시간 assert가 없다(약 334~354행). upstream 정렬이 깨지면 contiguous segment의 의미가 바뀐다. 이는 확인이 필요한 guard 위험이다.
- 전체 contiguous segment의 median/MAD로 변화량을 정규화한다(약 357~371행). 이상 구간이 segment의 큰 비율이면 이상 자체가 기준선을 끌어올려 장기 shift를 self-normalize할 수 있다.
- proposal probability가 `min(physical_change_score, 1-anchor_probability)`로 구성된다(약 475~487행). anchor 주변의 중간 확률 영역에서 rescue 후보가 오히려 억제될 수 있다.
- flank가 없을 때 feature를 실수 0으로 채우고, slope는 finite sample index를 압축해 계산한다(약 600~672행). “관측 없음”과 “실제 0 변화”가 섞이고, 내부 gap의 시간길이가 사라진다.
- proposal rescorer는 proposal multiplicity와 이벤트 길이를 반영한 sample weight 없이 fit한다(약 805~828행). 한 이벤트에서 많은 유사 proposal이 나오면 그 이벤트가 학습을 지배할 수 있다.
- decoder는 classifier probability threshold를 boundary z-score 변환값에도 동일하게 적용한다(약 914~915행). 두 값은 calibration이 다른 척도다.
- inner 54 fits 중 decoder만 다른 동일 입력·seed 재학습이 절반이다. 실제 unique model은 27개이므로 probability cache를 공유하면 계산을 절반 가까이 줄일 수 있다.

### 4.3 이미 확인된 모델 병목

- incumbent는 precision 0.9518에 recall 0.7922다. 추가 precision보다 장기 offset/drift의 누락행 복구가 우선이다.
- 최근 v1r6 candidate는 로컬 OOF 421,032행 전부에서 anchor와 같았고 rescue 행 0, `ΔF1 = 0`으로 `NO_GO_LOCAL_GATE`였다. threshold 미세조정보다 실제 추가행을 만드는 구조가 필요하다.
- 2026-08-13의 이전 XGBoost OOF anchor(F1 0.860371) 실패 재구성에서는 false negative의 92.7%가 offset/drift, 77%가 48시간 이상 이벤트에 모였다.
- 같은 이전 anchor의 17개 장기 이벤트 중 14개는 이미 적어도 한 행이 탐지돼 있었다. 이는 비파괴 확장 가설을 지지하지만 현재 Round-B anchor에서도 분포가 같다고 단정하지 않으며 Pre-cycle에서 재확인한다.
- hard typed semi-Markov는 `ΔF1 +0.00243`에 그쳤고 spike·최악 그룹을 훼손했다. generic TCN은 incumbent보다 크게 낮았고, density-ratio branch는 구조 gate를 통과하지 못했다.

## 5. P1 돌파구: F1-aware non-destructive interval rescue

### 5.1 구조

1. **Anchor 동결:** incumbent positive는 어떤 경우에도 지우지 않는다.
2. **Target-free proposal bank:** offset은 robust PELT-L1, drift 진입·이탈은 NOT 또는 piecewise-linear change detector를 사용한다. bank와 penalty 범위는 실행 전에 고정한다.
3. **0-fit oracle ceiling:** 사전등록된 pre-outer discovery/inner label에서만 proposal library의 최대 row-F1을 계산한다. Q2/Q3/Q4 outer label은 proposal bank·feature·decoder가 완전히 freeze되기 전에는 이 계산에 쓰지 않는다.
4. **F1 utility target:** 각 proposal에 대해 `ΔTP`, `ΔFP`, 실제 이벤트 coverage, anchor 연결성, 경계 대비를 산출한다. proposal overlap binary target은 폐기한다.
5. **작은 cross-fitted rescorer:** proposal이 새로 더하는 row 수 `n_new`를 exposure로 둔다. 1순위는 성공수 `ΔTP`, 실패수 `ΔFP`의 regularized binomial GLM이다. nonlinear 대조군은 두 Poisson head를 쓰되 비음수 raw prediction을 equality simplex에 투영해 `E[ΔTP]=n_new*p`, `E[ΔFP]=n_new*(1-p)`를 강제한다. 두 head는 각각 1 physical fit으로 센다.
6. **Non-overlap decode:** 먼저 calibration된 boundary 조건으로 eligible proposal을 고정한 다음, F1의 비가산성을 보존하는 linear-fractional DP를 푼다. 현재 F1 후보값 `lambda`에서 proposal weight를 `(2-lambda)E[ΔTP] - lambda E[ΔFP]`로 두고 weighted interval scheduling을 푼 뒤, anchor 혼동행렬의 exact F1로 `lambda`를 갱신하는 Dinkelbach 반복을 고정 tolerance까지 수행한다. boundary filter가 바뀌면 DP를 다시 풀며, 최종 outer 판정은 실제 row count의 exact F1로 한다.

Anchor의 기존 혼동행렬이 `(TP0, FP0, FN0)`이고 proposal이 `(ΔTP, ΔFP)`를 더하면,

`F1' = 2(TP0+ΔTP) / [2(TP0+ΔTP) + FP0+ΔFP + FN0-ΔTP]`

이다. 따라서 같은 80% purity label을 받은 proposal이라도 기여하는 TP·FP 수와 현재 anchor 상태에 따라 가치가 다르다. 이 차이가 현재 target이 놓치는 핵심이다.

### 5.2 외부 근거와 전이 범위

- PELT는 penalized segmentation의 정확해를 조건부 선형시간에 찾는다. P1의 긴 연속 시계열에 계산적으로 맞지만 penalty 오설정은 과·소분할을 만든다.
- NOT는 가장 좁은 유의 구간을 우선해 다중 변화의 혼합을 줄이고 piecewise-linear kink도 다룬다. drift의 진입·이탈 후보에 적합하지만 P1 drift event 수가 적어 bank를 넓히면 과적합된다.
- BSN은 boundary evidence와 proposal-level confidence를 분리한다. 여기서는 영상용 neural network를 가져오는 것이 아니라 **분리된 두 점수 구조**만 차용한다.
- F-measure 연구는 확률모델 학습과 metric-specific inference를 분리하는 plug-in 접근의 이론적 정합성을 제공한다. P1에서는 proposal score와 F1-aware interval selection을 분리하는 근거다.

### 5.3 사전등록 gate

- 0-fit oracle `ΔF1 < +0.0255`이면 수학적 hard NO-GO다. 불완전한 scorer의 감쇠 여유를 위해 full Cycle 1은 oracle `>= +0.040`일 때만 허용한다. `+0.0255~+0.040` 회색구간은 binomial GLM 1개만 저비용 진단하고 full bank/GBDT로 확장하지 않는다.
- 최종 pooled `ΔF1 < +0.0255` 또는 paired KST-day bootstrap CI90 하한 `< +0.012`이면 제출 금지.
- disconnected-interval precision `< 0.75`, FP/day ratio `> 1.05`, spike/flatline mutation, 3개 outer 중 2개 미만 개선 중 하나라도 발생하면 NO-GO.
- leave-one-proposal-family-out에서 효과가 소멸하면 proposal bank 우연으로 판정하고 종료.

위 항목은 핵심 요약이다. 실제 승격에는 improving stations 2/3, Q3·G-ORS 비악화, noise recall `>= -0.005`, long-event recall 개선, equal-weight supported-cell 개선을 포함한 기존 research gate 전부도 통과해야 한다. 단, 이 historical outer들은 과거 가설 탐색에 노출됐으므로 CI90은 fresh-holdout 신뢰구간이 아니라 **stability interval**로 해석한다.

## 6. P2 코드 감사: 표현력보다 수직구조와 수송성

### 6.1 최우선 검증 결함 — 비교기가 이미 평가면을 보았다

아직 미실행인 prospective final raw 설계는 `p2_extrapolated_soft_gate_v2`를 exact incumbent로 고정한다. 이 prospective 설계 자체의 physical fit은 0이다. 다만 비교기로 지정된 실행 lineage에는 `adaptive_after_outer_exposure=true`, `fresh_holdout_claimed=false`가 명시돼 있고, layer-2 factor 10은 2024년 9~10월 노출 OOF optimum 9.779를 반올림해 골랐다. 실행된 comparator의 실제 개선은 같은 9~10월에서 -0.001251°C에 불과했고 2025년 11~12월에는 +0.001257°C 악화했다.

따라서 이 비교기는 “현재 배포 CSV를 이기는가”에는 유효하지만 “새 family가 구조적으로 우월한가·운송 가능한가”의 단독 판정기준으로는 부적합하다. 새 후보는 다음 두 축에 동시에 비교해야 한다.

1. 실제 frozen deployed incumbent.
2. 같은 계산예산·같은 inner-only 선택을 쓴 architecture-matched base.

factor와 blend는 pre-outer inner block에서만 선택하고 prediction을 freeze한 뒤 model-fit-held-out outer에서 한 번 판정한다. 이 outer도 family 수준에서는 과거 가설에 노출됐음을 별도 표기한다. 기존 OOF의 leave-one-block-out factor 재점수는 comparator 편향 진단에만 쓰고 새 family 선택에는 쓰지 않는다.

### 6.2 미실행 prospective final-design의 구조적 위험

대상: `src/p2_restore/features.py`, `src/p2_restore/research.py`, `configs/experiments/p2_public_group_balanced_celsius_residual_v2_final_design.json`

- 등록된 55개 공개 feature는 주로 한 시점의 공개 T/S 단면, 깊이, 시간 harmonic이다. 1시간~7일의 causal 변화량·평균·분산과 수직 gradient 변화율이 없다.
- 등록된 학습범위는 2024년 5~6월 및 7월~8월 24일 두 window뿐이고 핵심 검증은 9~10월이다. 실행한다면 instantaneous feature만으로 계절 잔차 이동을 운송하기 어렵다.
- 등록된 fit objective는 layer × window × KST-day를 완전히 균등화하지만 공식 평가는 row-pooled RMSE다. 균등화는 robust guard로는 유용하지만 primary objective가 되면 작은 그룹의 고분산을 과대 반영할 수 있다.
- 등록된 핵심 판정은 61일 단일 surface이고 paired bootstrap도 KST-day 단위를 독립 재표집한다. 실행 전 3일·7일 moving-block stability interval을 추가한다. 세 historical season의 부호는 위험진단으로 보고하되, 사용자의 기존 결정에 따라 개별 계절의 소폭 악화를 자동 veto로 쓰지 않는다.
- training baseline은 nominal depth로 재계산하고 deploy feature path는 제공된 baseline을 받는다. 두 경로가 수치적으로 동일한지는 0-fit parity check가 필요하다. 현재 증거만으로 확정 버그라고 단정하지 않는다.
- prospective final target은 raw residual °C인데 representation은 dimensionless normalized-curvature target용 55 features를 그대로 쓴다. tree가 shape×scale 상호작용을 다시 근사할 수는 있지만, raw-°C vertical contrast와 absolute water-mass state를 금지한 채 400 trees에 이 상호작용을 떠넘길 위험이 있다. raw target이면 Celsius contrast/gradient·scale×shape 직접항을 허용하거나 normalized target→Celsius decode로 되돌려야 한다.
- 기존 naive EOF는 temperature-only global PCA rank 3, complete profile만 사용, 관측층 least-squares score 복원이다. season/covariate mean, T/S cross-covariance, measurement-error covariance, shrinkage가 없다. 그 실패는 conditional multivariate FPCA의 반증이 아니다.
- prospective LightGBM의 `subsample=0.85`에는 `subsample_freq/bagging_freq`가 없어 실행 시 row bagging이 비활성이다. 세 seed가 완전히 동일하지는 않지만 기대한 다양성보다 작다. `subsample_freq=1`을 명시하거나 subsample 설명을 제거해야 한다.

### 6.3 이미 소진된 경로

- naive EOF rank 3은 세 historical block에서 RMSE 약 1.921 / 8.057 / 2.794°C로 실패했다.
- 일반 선형 tide residual·RTS precheck는 모든 target layer aggregate R²가 음수였다.
- deep model은 일부 OOF 개선이 LOBO에서 크게 줄었고 full-prefix에서 방향이 뒤집혔다.
- hard physical projection은 약 -0.00124°C 수준에 그쳤고 structured mask imputer는 gate에서 거절됐다.
- 최근 P2 A/B 제출은 original보다 각각 +0.172435°C, +0.058836°C 악화했다. 더 큰 HPO가 아니라 validation transport를 바꿔야 한다.

## 7. P2 돌파구: shrinkage conditional hydrographic FPCA

### 7.1 구조

층 번호를 연도 간 고정 좌표로 풀링하지 않는다. 정적 감사에서 2024년 layer 7은 약 49m지만 2025년 layer 7은 약 39m이고, 2025년 layer 8이 약 49m인 depth-regime 변경을 확인했다. 따라서 각 outer fold `f`마다 nominal depth와 허용오차로 정렬한 `D_f`를 만들고 `z_f = [T(d): d in D_f, S(d): d in D_f]`를 구성한다. 2024 regime은 layer 1~6의 실제 수심과 약 49m, 2025 regime은 같은 수심대에 약 39m와 약 49m를 포함한다. 2024에 없는 약 39m는 결측치가 아니라 **구조적 부재**이며, 가짜 all-year 16변수 complete profile을 만들거나 layer 7끼리 직접 풀링하지 않는다. 목표는 기존 온도층 2·3·4로 유지한다. 각 outer의 관측 블록 `B_o`에는 그 outer의 pre-train과 배포면에 공통으로 실제 관측되는 public T/S node만 넣는다. 필요한 target 또는 observed node의 교차공분산이 pre-outer에서 식별되지 않으면 해당 fold와 FPCA 계열을 NO-GO로 판정한다.

각 physical-depth×variable의 계절 평균 `mu_j(t)`는 `[1, sin(2πd), cos(2πd), sin(4πd), cos(4πd)]`의 고정 K=2 Fourier OLS로 outer-train에서만 적합한다. 최소 60 KST day와 120일 span을 못 채우면 그 변수는 train-only global mean으로 fallback한다. 계절 평균을 뺀 뒤 각 변수는 outer-train SD `s_j`로 표준화하며 floor는 `1e-6`이다. 모든 FPCA 연산은 이 무단위 좌표에서 하고 목표 온도는 마지막에 `s_m`과 `mu_m(t)`로 역변환한다.

표준화 residual covariance `S`에는 사전고정 shrinkage `alpha`를 적용해 `Sigma_tilde = (1-alpha)S + alpha*diag(S)`를 만들고, 상위 `q`개 eigenvector `B_q`와 eigenvalue `Lambda_q`를 얻는다. 관측 공개층을 `o`, 숨은 온도층 2·3·4를 `m`이라 두면 conditional-score 복원은 다음과 같다.

`r_o = (z_o - mu_o) / s_o`

`a_hat = Lambda_q B_o' (B_o Lambda_q B_o' + Psi_o + epsilon I)^(-1) r_o`

`z_hat_m = mu_m + s_m * (B_m a_hat)`

- `Psi_o`는 q와 무관하게 한 번 추정한다. outer-train에서 gap이 해당 변수 median cadence의 2배 이하인 연속 residual 차분을 모아 표준화 좌표에서 `psi_j = clip(0.5*[1.4826*MAD(Δr_j)]^2, 1e-6, 0.25)`로 고정한다. 유효 pair가 100개 미만이면 `psi_j=0.01`로 fallback한다. `epsilon=1e-6`은 inversion용 고정 jitter다. covariance shrinkage와 measurement error를 같은 파라미터로 섞지 않는다.
- rank `q`는 inner fold에서만 `{2, 3, 4}` 중 선택한다.
- rank truncation이 없는 full shrinkage Gaussian conditional mean은 별도 대조군으로만 두고 FPCA와 한 모델처럼 혼용하지 않는다.
- 각 outer block마다 `mu(t)`, `S`, `Psi`, eigenbasis를 outer 시작 7일 이전 데이터만으로 다시 fit한다. rank·shrinkage·blend는 그 pre-outer 구간 내부의 chronological inner split에서만 선택하고, model-fit-held-out outer truth는 prediction freeze 뒤 한 번 평가한다. outer는 hypothesis-exposed surface이므로 CI는 stability interval이다.
- T/S를 함께 쓰되, 출력은 온도 2·3·4만 생성한다.
- 기존 depth-linear baseline과의 blend 계수는 nested OOF에서 한 번 고정한다.
- primary score는 row-pooled RMSE, layer-equal/day-equal은 guard로 함께 보고한다.

### 7.2 왜 기존 EOF와 다른가

| 기존 naive EOF | 제안 conditional hydrographic FPCA |
|---|---|
| 온도만 사용 | 온도+염분의 교차공분산 사용 |
| layer ordinal 중심 좌표 | nominal-depth regime별 fold-specific grid; 구조적 부재 분리 |
| 전기간 global mean | fold-train-only 계절 mean/covariate mean |
| 고정 rank 3 | inner-only rank 2/3/4 |
| 무정규화 least squares | measurement error + covariance shrinkage |
| 관측층에서 score 단순 투영 | 부분관측 조건부 score/conditional mean |
| 모델 단독 출력 | depth-linear anchor와 잠긴 blend |

2026년 conditional multivariate FPCA 연구는 완전한 이변량 T/S profile에서 평균·공분산·고유기저를 학습하고, measurement error를 포함해 부분 관측 profile의 PC score를 조건부 추정했다. 구조적으로 P2와 매우 가깝지만, 넓은 남빙양·깊은 꼬리 누락 사례이고 2026년 preprint이므로 P2에서의 효과는 직접 검증해야 한다.

TS-Cast와 최근 해양 profile 복원 연구도 climatology를 물리적 prior로 두고 동적 관측으로 anomaly를 조정하는 설계를 지지한다. 다만 15만 profile U-Net/Transformer를 복제하지 않고, 소표본 P2에 맞는 저차원 축소판만 취한다.

### 7.3 사전등록 gate

- P2의 보수적 stability upper는 paired-day, 3-day moving-block, 7-day moving-block 90% upper 중 최댓값으로 고정한다. 다일 block이 악화하면 veto한다.
- 공식 proxy와 같은 row-pooled RMSE를 primary로 유지하고, **각 target layer** RMSE delta는 기존 prospective gate `<= +0.003°C`를 적용한다.
- 개별 historical block은 위험진단으로 전부 보고하지만 소폭 악화를 자동 탈락으로 쓰지 않는다. worst-block `+0.020°C` 같은 새 catastrophic veto를 도입하려면 별도 사용자 승인을 받는다.
- layer 4가 개선되지 않으면 NO-GO.
- exact 9~10월 `ΔRMSE > -0.015°C` 또는 보수적 stability upper `> -0.005°C`이면 Cycle 2 금지.
- 제출 후보는 더 강한 기존 gate `ΔRMSE <= -0.060°C`, 보수적 stability upper `<= -0.040°C`를 통과해야 한다.
- external reanalysis는 규정 허용·grid/depth 대표성·동화 독립성이 모두 증명되기 전까지 사용하지 않는다.

## 8. Pre-cycle audit 이후 정확히 두 학습 사이클

### Pre-cycle 0 — 0-fit 계약 감사 + 고정 analytic screen (학습 사이클에 산입하지 않음)

**P1**

1. 저장된 outer p100 probability를 deployment 규칙 하나로 재decode하고 fold별 anchor/F1/proposal/connected-count 차이를 계산한다.
2. challenger training cutoff ledger가 outer start 이전이며 Q2<Q3<Q4로 row/time support가 증가하는지 확인한다.
3. input order·시간 단조·gap segmentation assert를 추가한 별도 연구 branch를 만든다.
4. 고정 proposal bank의 event recall, proposal multiplicity, length별 coverage를 계산한다.
5. anchor 보존 조건의 oracle interval set으로 최대 row-F1을 계산한다.
6. oracle `+0.0255` 미만이면 P1 proposal 계열을 종료하고, `+0.040` 미만이면 binomial GLM 1개 진단 외 full Cycle 1을 금지한다.

**P2**

1. 노출 OOF factor를 leave-one-block-out으로 재점수하고 pre-adaptive base와 동일 key에서 비교한다.
2. actual frozen incumbent와 architecture-matched base의 이중 comparator를 봉인한다.
3. training baseline과 deploy baseline의 동일 key parity를 검증한다.
4. raw-target/normalized-feature amplitude metamorphic test와 LightGBM bagging parameter test를 실행한다.
5. 각 outer에서 nominal-depth mapping receipt를 만들고 fold별 가용 변수 집합을 봉인한다. 2024는 layers 1~6의 실제 수심과 약 49m, 2025는 같은 수심대에 약 39m·49m를 포함해야 한다. 2024의 약 39m 부재는 missingness 계산에서 제외한다.
6. 봉인된 fold별 변수 집합에서만 완전 profile 수, 계절별 covariance condition number, public-to-hidden cross-covariance를 측정한다. target T2·T3·T4와 배포 공통 observed node 중 하나라도 식별 불가능하면 즉시 종료한다.
7. fold-specific complete-profile population과 deployment-like public-valid population의 month 분포와 공개층 T/S 분포를 비교한다. 모든 fit month에 complete profile 200개 이상, complete share 70% 이상, month total-variation distance 0.15 이하, 공개변수 최대 standardized mean difference 0.25 이하를 모두 요구한다.
8. representativeness gate가 실패하면 complete-case FPCA 계열을 종료한다. pairwise/EM covariance를 같은 결과를 보고 추가하지 않고 별도 사전등록 연구로 넘긴다.
9. 고정 `q=3`, 고정 shrinkage 1수준의 optimizer-free analytic FPCA를 final outer가 아닌 사전등록 pre-outer design/inner block에서만 1회 fit·score한다. 이 screen은 명시적으로 analytic fit으로 계상하며 hyperparameter 선택·blend를 금지한다.
10. 각 design validation에서 `fit end < validation start - 7 days`, validation truth의 fit 미사용, prediction freeze 뒤 score라는 chronology receipt를 남긴다. 세 final historical outer truth는 Cycle 1 one-shot까지 이 screen에 쓰지 않는다.
11. layer 4 identifiability가 없거나 design/inner pooled score가 anchor보다 나쁘면 FPCA 계열을 종료한다. 개별 design block 부호는 진단으로만 남긴다.

### Cycle 1 — 최소 모델 비교

**P1: 최대 6~18 fits**

- PELT-L1 + NOT/linear bank 고정.
- event/day-balanced regularized binomial GLM(호출당 1 physical fit)과 shallow Poisson-GBDT two-head(호출당 2 physical fits)만 비교한다. inner/outer의 모든 실제 fit 호출을 6~18 상한에 포함한다.
- inner fold에서 utility calibration·boundary gate를 잠그고 outer 3개 window에 그대로 적용.
- Cycle 1부터 동일 Dinkelbach non-overlap decoder를 사용한다. 결과에 따라 greedy/DP를 바꾸지 않는다.
- current 72-fit 계획은 폐기한다.

**P2: 최대 6~12 cells**

- fixed K=2 seasonal mean을 유지한 채 rank `{2,3,4}` × shrinkage 2수준만 비교한다. 계절 mean의 자유도·on/off는 결과 후 바꾸지 않는다.
- 3개 historical outer block 각각에서 mean/covariance/eigenbasis는 outer 이전 데이터만 사용하고 7일 purge한다. rank·shrinkage·blend는 pre-outer inner split에서만 선택한 뒤 prediction을 freeze한다. 이 outer는 model-fit-held-out이지만 hypothesis-exposed임을 결과에 표시한다.
- row-pooled RMSE를 primary로 하고 layer/day 균형지표와 3일·7일 moving-block CI를 transport guard로 둔다.
- CPU 병렬은 cell 수준에서만 사용하고 하나의 cell 내부 BLAS thread를 제한해 oversubscription을 막는다.

### Cycle 2 — Cycle 1이 gate를 통과할 때만

- P1: Cycle 1의 동일 Dinkelbach decoder를 유지하고 soft duration spline의 순증분만 검증한다. duration 순증분 `< +0.005 F1`이면 decoder 확장을 종료한다.
- P2: FPCA residual에 public T/S의 causal lag `1/6/24/168h`, 변화량, rolling mean/std, vertical-gradient rate를 추가한다. 작은 ridge/GAM과 정규화 LGBM만 비교한다.
- 결과를 본 뒤 feature·fold·weight·proposal bank를 바꾸지 않는다. 새 가설은 다음 연구 문서로 넘긴다.

### 계산·토큰 규율

- 학습·bootstrap·해시·표 생성은 로컬 CPU가 수행하고, LLM은 설계·코드 리뷰·결론 취합에만 쓴다.
- 한 사이클의 상한은 P1 18 fits + P2 12 cells다. 0-fit gate에서 실패하면 대부분의 계산을 쓰기 전에 종료한다.
- 각 cell은 config hash, source hash, split digest, metric, paired CI, failure reason을 receipt로 남긴다.

## 9. 제출 의사결정

남은 제출은 튜닝 세트가 아니다.

1. 두 문제 모두 formal gate를 통과하기 전에는 제출하지 않는다.
2. 첫 제출은 가장 강한 local champion만 사용한다.
3. 두 번째 제출은 같은 모델의 threshold/blend 변형이 아니라 구조적으로 다른 hedge만 사용한다.
4. 두 공개점수로 blend 비율이나 calibration을 역산하지 않는다.
5. 후보 매핑·CSV schema·행 순서·해시는 별도 독립 QA 후 제출한다.

## 10. 최종 우선순위와 중단 기준

| 우선순위 | 가설 | 예상 역할 | 즉시 중단 조건 |
|---:|---|---|---|
| 0A | P2 comparator de-bias | 거짓 탈락·과대신뢰 방지 | dual comparator 봉인 실패 |
| 0B | P1 anchor/postprocess parity | selection→outer→deploy 동일화 | uniform decode 재현 실패 |
| 1 | P1 F1-aware CPD proposal rescue | 가장 큰 점수 headroom 공략 | hard stop <+0.0255; full-fit stop <+0.040 |
| 2 | P2 conditional multivariate FPCA | 숨은 수층의 구조적 복원 | physical-depth mapping·식별성 또는 row-pooled gate 실패 |
| 3 | P1 OOF Platt + nested F1 threshold | 저비용 hedge | outer ΔF1 < +0.0161 또는 불안정 |
| 4 | P2 seasonal prior + causal dynamics | FPCA 통과 후 잔차 보정 | row-pooled·conservative upper gate 실패 |
| 보류 | external ocean reanalysis prior | 규정 허용 시만 별도 연구 | 독립성·대표성 설명 불가 |

**최종 판단:** 현 코드의 취약점을 고치지 않은 채 더 오래 튜닝하는 것은 비효율적이다. 다음 실행은 P2 comparator de-bias와 P1 postprocess parity, 이어서 P1 proposal oracle과 P2 conditional covariance identifiability 순서다. 이 네 검사가 통과할 때만 작은 모델을 학습한다.

## 11. 한계

- 공식 점수는 사용자 제공 receipt와 2026-08-26 보관 스냅샷을 사용했다. 이 연구 중 live leaderboard나 공식 제출 파일을 다시 열지 않았다.
- local OOF와 official score는 평가 표본이 달라 절대 차이를 calibration 상수로 쓸 수 없다.
- P1 변화점 문헌은 proposal 생성 메커니즘을 지지하지만 P1의 최종 F1 개선을 보장하지 않는다.
- P2 conditional FPCA의 가장 직접적인 논문은 2026년 preprint이며, 다른 해역·샘플링 구조에서 평가됐다. P2는 2024/2025 depth regime이 달라 실제 수심 매핑과 fold별 공분산 식별성을 통과하기 전에는 방법론 적합성만 있는 조건부 가설이다.
- 점수 환산은 보관된 반올림 점수로 얻은 계획용 선형 근사다.

## 12. Claim-source ledger

| ID | 핵심 주장 | 내부 근거 | 외부 1차 근거 | 신뢰도 |
|---|---|---|---|---|
| C1 | P1 병목은 장기 offset/drift recall | P1 failure reconstruction, v1r6 QA | 없음; 로컬 직접 관측 | 높음 |
| C2 | 현재 proposal label/threshold는 row-F1과 불일치 | P1 source lines 675~719, 831~850 | Dembczynski et al. 2013; Lipton et al. 2014 | 높음 |
| C3 | PELT/NOT proposal bank는 계산·구조상 타당 | long-event segmentation source | Killick et al. 2012; Baranowski et al. 2019 | 중상 |
| C4 | anchor-preserving interval utility가 현 v1보다 낫다 | 코드 감사 + 실패유형 | BSN 2018 구조; F1 plug-in 이론 | 중상; 아직 미실행 |
| C5 | P2 병목은 표현력보다 시간수송성 | matched-budget compare, deep/LOBO reports | Ladder는 검증규율만 지지 | 높음 |
| C6 | naive EOF 실패는 conditional multivariate FPCA를 반증하지 않음 | research.py 및 method scout | Fonvieille et al. 2026 | 중상 |
| C7 | 계절 prior + 동적 anomaly 보정은 해양 복원에서 유효한 구조 | P2 feature gap | TS-Cast 2026; Liu et al. 2026 | 중상 |
| C8 | 반복 public 적응은 제출 과적합 위험 | A/B direction evidence | Blum & Hardt 2015 | 높음 |
| C9 | P2 현 exact comparator는 family 판정에 편향됨 | extrapolated soft-gate v2 config/result | 외부 근거 불필요; lineage 직접 증거 | 높음 |
| C10 | P1 selection·outer·deploy anchor가 다름 | learning-curve config, v3 amendment, v6 execution | 외부 근거 불필요; 코드 직접 증거 | 높음 |
| C11 | P2 layer ordinal은 연도 간 동일 실제 수심이 아님 | research.py 및 P2 reconnaissance의 nominal-depth ledger | 외부 근거 불필요; 데이터 계약 직접 증거 | 높음 |

## 13. 주요 1차 출처

1. Killick, Fearnhead & Eckley (2012), *Optimal Detection of Changepoints With a Linear Computational Cost*. [원문](https://arxiv.org/abs/1101.1438)
2. Baranowski, Chen & Fryzlewicz (2019), *Narrowest-Over-Threshold Detection of Multiple Change-points and Change-point-like Features*. [원문](https://arxiv.org/abs/1609.00293)
3. Lin et al. (2018), *BSN: Boundary Sensitive Network for Temporal Action Proposal Generation*. [원문](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Tianwei_Lin_BSN_Boundary_Sensitive_ECCV_2018_paper.pdf)
4. Dembczynski et al. (2013), *Optimizing the F-Measure in Multi-Label Classification*. [원문](https://proceedings.mlr.press/v28/dembczynski13.html)
5. Lipton, Elkan & Narayanaswamy (2014), *Thresholding Classifiers to Maximize F1 Score*. [원문](https://arxiv.org/abs/1402.1892)
6. Fonvieille et al. (2026), *Conditional multivariate functional PCA for reconstruction of temperature and salinity profiles partially sampled by deep-diving marine mammals*. [원문](https://arxiv.org/abs/2608.05376)
7. Chae, Donohue & Park (2026), *TS-Cast*. [원문](https://os.copernicus.org/articles/22/2161/2026/os-22-2161-2026.html)
8. Liu et al. (2026), *Data Driven Reconstruction of Upper Ocean Profiles*. [원문](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005679)
9. Blum & Hardt (2015), *The Ladder*. [원문](https://proceedings.mlr.press/v37/blum15.html)

## 14. 내부 증거 위치

- `reports/p1_v1r6_independent_postexecution_qa_20260826.json`
- `reports/p1_v1r6_independent_postexecution_qa_20260826.md`
- `reports/P1_FAILURE_RECON_2026-08-13.md`
- `reports/P1_ACADEMIC_METHODS_SCOUT_2026-08-13.md`
- `src/p1_qc/long_event_segment_proposal_rescore.py`
- `artifacts/p1_typed_duration_semimarkov_v2/result.json`
- `artifacts/p1_station_layer_temporal_convolution_event_v2/result.json`
- `artifacts/p1_target_covariate_density_ratio_xgb_v1/result.json`
- `src/p2_restore/features.py`
- `src/p2_restore/research.py`
- `artifacts/p2_method_scout/result.json`
- `artifacts/p2_tide_rts_v1/result.json`
- `configs/experiments/p2_extrapolated_soft_gate_v2.json`
- `artifacts/p2_extrapolated_soft_gate_v2/result.json`
- `configs/p1_meaningful_learning_curve_generation_v1.json`
- `configs/experiments/p1_long_event_segment_proposal_rescore_v3_execution_closure_amendment.json`
- `src/p1_qc/long_event_segment_proposal_rescore_execution_v6.py`
- `artifacts/matched_budget_local_compare_20260825/synthesis.json`
- `reports/leaderboard_gap_research_20260826_v2/deep_research_handoff_ko.md`

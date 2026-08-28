# P1·P2·P3 병렬 돌파구 Deep Research 및 실행

작성일: 2026-08-28 KST
상태: P1 NO_GO, P2 HOLD, P3 NO_GO
공식 업로드: 0건

## 결론

세 문제를 기존 실패·공식 점수·OOF 예측 벡터까지 다시 추적한 결과, 구조를 더 크게 만드는 것보다 **기존 champion과 다른 정보가 실제로 남아 있는지 분리하는 것**이 핵심이었다.

- **P1:** 기존 공식 성공 계보 안에서 `I-ORS/layer 5/e125=1/e150=0/Router=0` 셀이 Q2·Q4 두 drift 사건에서 pooled 44/44 TP였지만, full-data 3-seed exact replay에서는 추가 양성이 단 1행·연결성분 길이 1로 축소됐다. e150 예측 배열과 decode의 exact replay는 통과했으나 8–80행·성분당 8행 gate를 실패해 **NO_GO_EXACT_NO_OUTPUT**으로 종료했다.
- **P2:** 공식 alpha 0.50 점수를 포함한 RMSE² 반응면은 alpha 0.725에서 RMSE 약 0.4189를 예측했다. 그러나 실제 PAVA 투영 벡터를 다섯 공식 벡터의 span으로 검증한 강건 상한은 `0.435846`으로, 사전 gate `0.421252`를 통과하지 못했다. 결과를 본 뒤 gate를 낮추지 않고 **HOLD**한다.
- **P3:** ERA5·KMA·energy state-space·sequence checkpoint의 겉보기 개선에서 공식 champion의 alpha 축을 제거했다. 남은 직교 이득은 최대 ERA5 `-0.003331m`였으나 한 window가 악화했고, 같은 ERA5/Hs² family는 이미 공식에서 실패했다. 나머지는 CI가 0을 가로질러 **전체 NO_GO**다.

## 공통 질문과 연구 설계

이번 사이클의 질문은 “새 모델이 로컬에서 좋아졌는가”가 아니라 다음 세 가지였다.

1. 기존 공식 champion과 다른 prediction direction인가?
2. 그 방향이 사전등록 historical window·station·lead에서 같은 부호로 재현되는가?
3. 공식 점수를 사용한 축이라면 새 점수로 재계산한 강건 gate를 통과하는가?

반복 leaderboard 적응은 holdout 과적합을 만들 수 있으므로 한 축당 한 번의 고정 실험과 중단조건을 사용했다. 숨은 정답·공식 test label·외부 exact-source matching은 사용하지 않았다.

## P1 — checkpoint disagreement cell

### 발견 근거

현 공식 최고는 `MSTCN e150 Router union all`, Public F1 `0.833248`이다. 같은 제출 batch에서 G·S-only는 `0.822488`였고 두 파일은 I-ORS 추가 80행만 달랐다. 따라서 I-ORS의 고정 e150 추가 방향은 공식에서 양의 증거를 가진다.

OOF에서 frozen width 512, threshold 0.8, 3 seeds의 epoch 125와 150을 비교했다.

| 셀 | Q2 | Q3 | Q4 | pooled |
|---|---:|---:|---:|---:|
| I-ORS/L5, e125=1, e150=0, Router=0 | 8/8 TP | 0행 | 36/36 TP | 44/44 TP |
| 독립 사건 | drift 1개 | 0 | drift 1개 | 2개 |
| e150 대비 ΔF1 | +0.000841 | 0 | +0.004767 | +0.001559 |

반면 전역 e125 union은 349행 중 108 TP, precision `0.30946`, ΔF1 `-0.003035`였다. 따라서 epoch 125 전체는 닫고 위 셀만 one-shot으로 남겼다. 사건 단위 표본은 2개뿐이므로 2/2 exact 95% 하한 `0.1581`은 현재 F1을 올리는 추가 precision 문턱보다 낮다. 이는 저위험 exploit이 아니라 고위험·고정보 probe다.

### 실행 계약

- full data 3 seeds를 epoch 150까지 정확히 한 번 재생하고 epoch 125를 동시에 캡처한다.
- 새 epoch-150 continuous prediction 배열과 decode가 기존 full e150 artifact와 bit-exact여야 한다.
- 현 champion 양성을 모두 보존하고 I-ORS/layer 5/e125-only 셀만 OR 한다.
- 추가행 8–80, 연결성분 1–4, 모든 성분 길이 8 이상을 요구한다.
- 삭제·다른 cell·threshold·smoothing·checkpoint 재탐색은 금지한다.

실제 full-data 실행 결과는 3개 seed 모두 epoch 150까지 완료됐고, 기존 e150 continuous prediction과 decode를 정확히 재현했다. 그러나 고정 셀의 추가 양성은 1행뿐이었으며 연결성분도 길이 1 하나였다. champion 양성 6,394행은 전부 보존됐지만 후보는 6,395행이 되어 사전등록한 `추가 8–80행`과 `모든 성분 길이 8 이상`을 통과하지 못했다. 따라서 CSV·ready artifact를 만들지 않았다. 회고 44/44는 새 test support가 아니라 두 historical 사건에 국한된 발견이었다.

실행 코드: `scripts/run_p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1.py`
설정: `configs/experiments/p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1.json`
결과: `artifacts/p1_mstcn_e125_only_iors_l5_drift_rescue_20260828_v1/result.json`

Retrospective 재현 코드: `scripts/audit_p1_checkpoint_disagreement_cell_20260828.py`
재현 결과: `p1_checkpoint_disagreement_retroaudit.json`

## P2 — alpha50 이후 공식 손실기하

공식 RMSE 계보는 alpha `0/.1/.2/.4/.5`에서 각각 `0.535727/0.507628/0.483661/0.445147/0.431252`다. RMSE² 이차 반응면은 다음을 보였다.

| 적합 점 | 최적 alpha | 예상 RMSE |
|---|---:|---:|
| 전체 5점 | 0.725184 | 0.418871 |
| 최근 4점 | 0.764191 | 0.415359 |
| 최근 3점 | 0.769986 | 0.414859 |

하지만 alpha 0.725의 실제 투영 prediction direction을 다섯 scored vector로 분해하면 orthogonal residual RMS가 `0.019817`, 전체 방향의 `6.154%`였다. 숨은 reference error의 이 성분은 aggregate score만으로 식별되지 않는다.

- 중심 RMSE: `0.416624`
- 6자리 반올림 강건 구간: `[0.396471, 0.435846]`
- 사전 승격 상한: `0.421252`
- 판정: **gate FAIL, CSV 미생성, alpha 축 HOLD**

이 결론은 이차 반응면이 틀렸다는 뜻이 아니라, 남은 한 번을 쓸 만큼 강건하게 확인되지 않았다는 뜻이다. Public subset 구성과 Private transport가 미공개인 상황에서 결과 후 gate 완화는 하지 않는다.

실행 코드: `scripts/analyze_p2_official_metric_geometry_alpha50_20260828.py`
결과: `p2_metric_geometry_after_alpha50.json`

## P3 — champion alpha 축을 제거한 prediction basis 감사

고정 181사례·1,086행 historical OOF에서 18/24h만 weight 0.25로 혼합했다. 먼저 단순 basis-champion 방향을 평가한 뒤, target을 사용하지 않고 `A-O` champion alpha 축을 Gram–Schmidt로 제거했다. bootstrap은 case cluster, seed 20260828, 5,000회다.

| basis | support | 단순 ΔRMSE | 직교 ΔRMSE | CI90 | 최종 |
|---|---:|---:|---:|---:|---|
| ERA5 | 181/181 | -0.012296 | -0.003331 | [-0.005874, -0.000909] | same-family 및 window gate FAIL |
| KMA | 179/181 | -0.008934 | -0.001095 | [-0.005481, +0.003353] | FAIL |
| energy state-space | 181/181 | -0.014872 | -0.001645 | [-0.004839, +0.001821] | FAIL |
| sequence checkpoint | 181/181 | -0.011993 | -0.001278 | [-0.004396, +0.001785] | FAIL |

ERA5와 KMA prediction direction 상관은 `0.837934`였다. 이름이 달라도 상당 부분 같은 수축 방향이다. KMA는 권리·누출 gate를 통과했지만 직교 후 S-ORS `+0.006475m`, winter window `+0.009773m` 악화 및 CI 교차로 탈락한다.

실행 코드: `scripts/audit_p3_champion_orthogonal_prediction_basis_20260828.py`
결과: `p3_orthogonal_basis_audit.json`

## 승격 기준

- 로컬 개선의 크기보다 **prediction direction 독립성**, fold/window/station 일관성, confidence bound를 우선한다.
- P1처럼 공식 성공 계보의 작은 cell은 방향 증거와 exact replay를 요구한다.
- P2처럼 공식 반응곡선을 쓰는 경우 후보 vector 자체의 식별 불확실성을 포함한다.
- P3처럼 공식에서 한 번 실패한 family는 같은 family의 작은 OOF 개선으로 재개하지 않는다.
- 하루 3회는 상한이지 사용 목표가 아니다. 한 축의 결과를 보고 같은 날 미세조정 제출을 연속하지 않는다.

## 한계

- P1 subgroup은 retrospective하게 발견됐고 독립 사건이 2개뿐이다.
- P2 기하는 현재 Public scoring set 조건부이며 Private/final 운반을 보장하지 않는다.
- P3 historical OOF와 anonymous official cases 사이에는 큰 domain shift가 있다.
- 공식 점수는 행별 label을 알려주지 않으며, 본 연구도 행별 hidden label을 추론하지 않았다.

## 1차 출처

- Lipton, Elkan, Naryanaswamy (2014), *Optimal Thresholding of Classifiers to Maximize F1 Measure*: https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/
- Blum & Hardt (2015), *The Ladder*: https://proceedings.mlr.press/v37/blum15.html
- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*: https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
- Boyd & Vandenberghe (2004), *Convex Optimization*: https://web.stanford.edu/~boyd/cvxbook/
- Bates & Granger (1969), *The Combination of Forecasts*: https://doi.org/10.1057/jors.1969.103
- Ben-David et al. (2010), *A Theory of Learning from Different Domains*: https://doi.org/10.1007/s10994-009-5152-4

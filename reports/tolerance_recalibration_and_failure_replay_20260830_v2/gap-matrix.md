# 허용치와 과거 실패 재평가 gap matrix

| 요구 | P1 | P2 | P3 | 현재 결정 |
|---|---|---|---|---|
| 공식형 pooled primary | row micro-F1 | all-row temperature RMSE | all-row six-lead Hs RMSE | 효능 primary는 하나만 둔다. |
| 공식 score 식 | 경험 OLS만 있음 | 경험 OLS만 있음 | README anchor로 현재 구간 산출 가능 | P1/P2 point 환산은 planning-only. |
| 비영점 SESOI/비열등성 비용 | 없음 | 없음 | 없음 | directional margin `0`; equivalence margin 미설정. |
| numerical replay tolerance | exact/hash + `1e-12` | exact/hash + `1e-12` | exact/hash + `1e-12` | effect-size gate와 분리. |
| fresh unexposed local surface | Q2–Q4 반복 노출 | 3 historical windows 반복 노출 | 157/181 case 표면 반복 노출 | favorable result도 research-only. |
| dependence-aware interval | event+joint-day 필요 | contiguous KST-day+layers 필요 | episode 또는 contiguous day+six leads 필요 | row-IID CI 사용 금지. |
| selection/HPO uncertainty | Sobol 32-way | 243/729 evaluations | HPO confirmation reversal 존재 | untouched confirmation으로만 해소. |
| slice의 hard 목적 근거 | 미확인 | 미확인 | 미확인 | transport diagnostic; 안전/비용이 명시될 때만 hard. |
| local→official transport | GI는 일부 양성, 계열별 불안정 | rank-1 양성이나 surrogate 역전 | reverse-axis·ERA5에서 부호 역전 | 전역 scalar 금지. |
| outlier ground truth | 없음 | 없음 | 없음 | hard delete 기본값 금지. |
| 이번 official action | 미승인 | 미승인 | 미승인 | CSV/upload 0. |

## 새 상태별 대표 후보

| 상태 | P1 | P2 | P3 |
|---|---|---|---|
| high-value challenger research-only | 없음 | rank-1 base/cross-fit, nested PLS, Gaussian, state copula | 없음 |
| exploratory/reopen frozen confirmation | block, peer, balanced replay, segment router, window phase, Sobol trial18 | 없음 | lead-continuous |
| inconclusive | depth invariance, add-only LCB | density/addon 등 CI-cross families | KMA deployment, episode analog, sparse GP |
| primary harm exact recipe | Group-DRO, SupCon, target quantile 등 | availability v2, transfer/analog harms | CatBoost v3, SSL, RevIN, dense72 등 |
| invalid/no science | implementation/preflight failures | Gaussian v1, time-unit/inner-row failures | CatBoost v1/v2 technical, guard/support failures |

## 해소되지 않은 핵심 gap

1. P1/P2 공식 점수식과 Private mixture는 없다. 경험 OLS를 공식 허용치로 승격하지 않는다.
2. P1의 새 독립 labeled surface, P2의 untouched same-season surface, P3의 fresh storm episodes가 없다.
3. 여러 family가 동일 표면을 사용한 전체 family-selection bias는 현재 aggregate CI에 포함되지 않는다.
4. 제출 EV 계산에는 남은 날짜·quota, 후보별 Public→Private transport prior, 순위 효용이 필요하다. 따라서 숫자 하나로 모든 후보를 자르지 않는다.
5. 센서 오류 ground truth가 없어 outlier hard delete의 이득과 bias를 검증할 수 없다.

## 재평가 범위 해석

48-family ledger는 넓은 과학 계보 단위이고 35-group negative registry는 이후 exact 결과를 묶은 운영 단위다. 동일 실험이 양쪽에 있을 수 있으므로 `83개 독립 실험`이라고 해석하지 않는다. `failure-replay.json`은 두 목록을 각각 완전하게 보존해 누락 없이 다시 판정한다.

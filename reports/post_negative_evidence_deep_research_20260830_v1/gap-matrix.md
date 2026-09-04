# 실패축–새 실험 gap matrix

| 문제 | 이미 닫힌 정확한 축 | 남은 구조적 gap | 다음 실험 | 사전 preflight | kill gate | 최대 계획 자원 |
|---|---|---|---|---|---|---|
| P3 | valid CatBoost 138-fit selection 재탐색, Ordered+Depthwise invalid 조합, KMA alpha transport | confirmation projection의 schema contract 부재 | frozen `challenger_21` confirmation-only repair | required columns/order/dtype/key/hash end-to-end fixture | schema 불일치 또는 confirmation window 하나라도 비개선 | 0-fit contract + 최대 6 historical fits |
| P2 | PLS/OAS/GP/bridge/linear rank1·heave 계열 | 단조 비선형 marginal과 rank dependence를 함께 쓰는 원척도 conditional mean 미검증 | nonparanormal Gaussian-copula residual correction | train-only support, monotonicity, PSD/condition number, lineage | pooled 비개선, 2/3 fold 미달, layer >0.001°C 악화 | 3 outer folds, 작은 고정 shrinkage set |
| P1 | BCE+전역 threshold, rule/veto, synthetic injection, Group-DRO, 동일 Sobol space | 실제 event 단위 representation과 F1 학습목적 정렬 미검증 | event-balanced SupCon + calibrated F1 proposal head | event-disjoint split, class/event counts, anchor immutability | 어느 window든 ΔF1≤0, precision gate 실패, anchor removal>0, station share>0.8 | 1 seed 20–30 epoch screen; 후속 3-seed는 별도 승인 |

## 공통 차단 장치

- 공식 test/sample/submission/hidden label 접근 0
- 결과 기반 search space, fold, gate 변경 금지
- incumbent lineage hash 불일치 시 fit 전 중단
- 정확한 experiment ID별 one-shot lock
- terminal result에 fit count, runtime, config hash, data-access ledger 기록

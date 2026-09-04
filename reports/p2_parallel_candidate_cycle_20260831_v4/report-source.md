# P2 parallel candidate cycle 20260831 v4

## 결론

상태: `COMPLETE_WITH_NEW_INTERNAL_PASS_SUBMISSIONS_NOT_UPLOADED`. 내부 PASS 수: 1; 제출본 생성 수: 1.

기존 v3 shallow PASS까지 합치면 현재 P2는 내부 게이트를 통과한 서로 다른 제출본 2개를 보유한다. 새 HGB는 pooled 개선량이 더 크지만 Jul-Aug 회귀도 더 크므로 계절 운반 위험이 높은 정보성 후보다. 기존 shallow은 pooled 개선량은 작지만 Jul-Aug 회귀가 상대적으로 작다.

## 사전 고정 방법과 결과

| 후보 | 구조 | pooled delta RMSE | Sep-Oct | Jul-Aug | Nov-Dec | gate |
|---|---|---:|---:|---:|---:|---|
| P2_2_HGB_ABSOLUTE_PROFILE | hgb_absolute | -0.015491434 | -0.036017564 | +0.039669015 | -0.000915472 | PASS |
| P2_3_PLS2_ROBUST_PROFILE | pls2_profile | -0.003465139 | -0.009287113 | +0.009969101 | +0.025584690 | FAIL |

두 후보는 결과 확인 전 코드 상수로 고정했다. HGB 후보는 train-only absolute loss로 이상값 영향만 완화하며 행을 삭제하지 않는다. PLS 후보는 세 target layer를 공동 low-rank profile로 학습하고 train-only median imputation, 0.05 C profile RMS cap, 0.15 C absolute cap을 적용한다.

PASS 기준은 pooled RMSE 개선, 3개 blocked fold 중 최소 2개 개선, 공식 유사 2024 Sep-Oct 개선의 논리곱이다. 공식 test_index는 모든 내부 prediction hash가 봉인되고 점수와 gate가 끝난 뒤 PASS가 있을 때만 읽었다.

## QA와 경계

실제 fit count: 15. 내부 prediction hash 2개를 기록했다. hidden truth 접근과 upload는 0건이다. 제출 CSV는 PASS 후보만 만들었다.

첫 실행은 sklearn HGB가 fold 내 all-missing feature를 binning하면서 prediction/metric 전에 기술 종료됐다. 기존 `attempt_lock.json`을 보존하고 `technical_recovery.json`에 성능 미열람, prediction 0, official read 0을 기록한 뒤 train fold에서 finite/nonconstant feature만 선택하는 스키마 수리만 적용했다. 후보, 파라미터, split, correction, gate는 바꾸지 않았다. 초기 runner hash는 `56c9fbfd6ca51c5c7a935ccf60065d87533439f53040e732b0fc488dee1b3a47`, 수리 runner hash는 `cb773cd8e6ad34b0b111b8f5b67603bde3db226929d5aac185a65fad34019b42`이다. 독립 재계산 QA는 recovery chain, 두 prediction hash, pooled/fold metric, gate, 제출 구조/PAVA를 모두 PASS로 확인했다.

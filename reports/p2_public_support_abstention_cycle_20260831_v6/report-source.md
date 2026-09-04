# P2 support-abstention cycle 20260831 v6

## 결론

`COMPLETE_NO_PASS`. 층별 logistic support gate가 Oct ΔRMSE -0.011503661°C까지 개선하고 계절 불일치 구간을 exact abstain했지만, raw 기대 +0.111971점에서 수송 penalty를 차감한 값은 -0.009711점이라 `>=+0.01`을 통과하지 못했다. HGB gate도 calibrated -0.066267점으로 탈락했다. 4 fits, 공식/hidden/CSV/upload 모두 0이다.

이 실패를 threshold 결과맞춤으로 재시도하지 않고, 다음 v7에서 public hydrographic lag/state 정보를 추가하는 구조적 변경으로 이어갔다.

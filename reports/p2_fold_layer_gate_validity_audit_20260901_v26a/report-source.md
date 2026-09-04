# P2 v20-v26 fold×layer gate validity audit

## 결론

기존 aggregate safety gate는 국소 harm을 숨겼다. 실행된 6개 후보 중 v24만 9/9 fold×layer cell이 모두 non-harm이었다. v20, v21, v23, v26은 6/9만 non-harm이고 최대 cell 악화가 각각 +0.030831, +0.042112, +0.026044, +0.028924 C였다. v25도 7/9이며 최대 +0.007908 C였다.

이 감사는 과거 terminal 결론이나 현재 리더를 사후 변경하지 않는다. 앞으로만 기존 pooled/fold/month/aggregate-layer/CI/points gate에 다음을 추가한다.

- 9개 fold×layer cell 중 최소 8개는 ΔRMSE ≤ 0이어야 한다.
- 어떤 cell도 ΔRMSE가 +0.003 C를 넘지 않아야 한다.
- 실패 시 router, blend 조정 또는 threshold 완화 없이 `SAFETY_NOT_READY`로 두고 genuinely fresh confirmation만 허용한다.

0.003 C는 이번 결과에서 새로 고른 값이 아니라 이미 preregistered month/layer tolerance를 동일한 국소 단위로 확장한 것이다. coverage 조건은 한 fold의 큰 개선이 다른 fold-layer 악화를 가리는 문제를 별도로 막는다.

이번 작업은 기존 result JSON 영수증만 read-only로 사용한 0-fit 감사다. observations/official/query/hidden/CSV/upload 접근은 모두 0이다.

# P2 v50 masked third-central-moment profile-pooling DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_MASKED_THIRD_CENTRAL_MOMENT_PROFILE_POOLING`. pooled delta RMSE `-0.051460125 C`, canonical nominal `+0.645698` points, transport `+0.524015` points.

prospective fold x layer gate `False`, non-harm `6/9`, max cell `+0.030862566 C`.

Exact v13 masked mean/max summary에 masked signed third central moment 32개를 추가했다. 새 head columns는 zero-init되어 initial function이 v13과 같다. 두 READY preflight는 byte-identical이고 상속 source-contract 5개 key의 synthetic missing-key rejection을 lock 전에 검증했다. 배포 observations.csv 및 그 truth-free 파생 training frame 외 자료, pretrained weights, official/test/sample/baseline/query/hidden/CSV/upload는 사용하지 않았다.

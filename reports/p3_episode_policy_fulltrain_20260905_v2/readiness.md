# P3 조건부 전체학습 경로: 준비만 완료, 실행하지 않음

상태: **NOT_EXECUTED_NO_INTERNAL_WINNER**. P3-B의 기존 clean no-op이 승자이므로 root의 조건부 full4+router1 승인 조건이 충족되지 않았다. 전체학습/추론/CSV/upload0, attempt lock과 model output 폴더는 생성하지 않았다.

전용 runner/config는 적격 B 승자가 legacy보다 엄격히 낮은 pooled RMSE이고 matching result SHA의 independent QA PASS가 있을 때만 full fitting을 허용하도록 준비했다. 현재 기록으로 `--mode train`을 호출하면 fitting 전에 차단된다. 이 문서를 새 실험 승인이나 기존 run 재개 허가로 해석하지 않는다.

- [config](../../configs/experiments/p3_episode_policy_fulltrain_20260905_v2.json), [runner](../../scripts/run_p3_episode_policy_fulltrain_20260905_v2.py), [tests](../../tests/test_p3_episode_policy_fulltrain_20260905_v2.py).
- synthetic inference tests5 PASS, Ruff PASS. Two-seed 성분별 clip 후 평균, label-free router 입력, short no-op, shrink0.2, target 변조 불변, case key 중복 차단을 검증했다.
- 준비된 runner SHA-256 `2d6f174decb924fe85b45e9f4e02143abd1be52f621b2405c145b3bef39aad86`, config SHA-256 `55293faec6a9605c4e7ab5b7e07fe257630dc65339f9363800617c0a8bc3901e`.
- 만약 미래의 다른 실험에 활용하려면 새 ID·승인·사전등록이 필요하다. 현재 B/A artifact나 gate를 사후 바꾸지 않는다.

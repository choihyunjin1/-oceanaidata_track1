# C3 빈 모델 디렉터리 재생성 — 자체 산출물 hash 가드 기술 실패

상태: **TERMINAL_TECHNICAL_FAILURE**, 성능 실패가 아니다. 기존 산출물/lock/seal을 보존했고 자동 재시작하지 않았다.

`RUN_TRAINING`은 비어 있는 새 `artifacts/p2_clean_regeneration_20260905_v4/03_model/`에서 seed 20260901의 60 epochs를 실제 scratch 학습했다. 첫 fit은 19.859초, 당시 전체 35.375초였다. 새 `model_seed20260901.pt` 저장 뒤 canonical trainer가 SHA-256을 계산하려 열었으나 새 adapter의 `path_allowed(... RUN_TRAINING, writing=False)`가 모든 `.pt` 읽기를 막아 중단됐다.

이는 기존 모델/답안을 학습에 사용한 문제가 아니라 **방금 생성한 산출물의 무결성 hash 읽기와 모델 로딩을 동일하게 차단한 guard 오류**다. 아직 3seed 완료·fresh-process 추론·26,061행 답안 생성·기준 재생성 PASS가 아니다. 학습 중 공식 키/샘플값·기존 모델·답안·OOF 접근은 0, CSV/upload 0이다. guard는 source 외 CSV 및 old/new model reads를 막았으며 OS 전체 감사를 의미하지는 않는다.

합성 pytest 4 PASS/Ruff PASS가 이 오류를 막지 못했다: allowlist 테스트가 신규 모델 hash 읽기를 필요 동작으로 검증하지 않았기 때문이다. 기술 정정 시 **신규 실행 자신의 모델 hash 읽기 허용 + `torch.load` 자체를 training 단계에서 금지**해야 한다. 별도 새 ID/승인을 요청했고 기존 lock 재사용/아티팩트 덮어쓰기는 하지 않았다.

근거: [runner](../../scripts/run_p2_clean_regeneration_20260905_v4.py), [seal](preregistration-seal.json), `artifacts/p2_clean_regeneration_20260905_v4/03_model/progress.json`, 실패 traceback은 실행 세션 12907. GPU 프로세스는 종료되어 root/P3에 해제 통보했다.

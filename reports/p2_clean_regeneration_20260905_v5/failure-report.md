# v5 — native writer와 Python 감사 이벤트 차이로 기술 종료

상태는 `TERMINAL_TECHNICAL_FAILURE`이며 모델 성능 실패가 아니다. v4의 금지 규칙을 단순 write-event registry로 정정했지만, PyTorch의 native 저장 경로가 Python `open` audit 이벤트를 만들지 않아 새 모델 경로를 registry에 등록하지 못했다. 첫 seed 60epoch의 실제 학습과 저장 뒤 SHA 확인에서 다시 막혔다.

v5의 pure path 합성 테스트는 PASS였으나 실제 native `torch.save`까지 실행하지 않아 이 결함을 발견하지 못했다. 실패 원인은 테스트 범위 누락이며 기존 모델·답안 입력이나 데이터 누출이 아니다. v5 runner/config/seal/lock/새 첫 모델/실패영수증은 보존한다. 자동 재시작하지 않았고 새 v6 기술정정 승인을 별도로 받았다.

v6에서는 실제 별도 subprocess와 active audit hook 아래 3개 native 모델 저장→자기 SHA→다음 저장을 모두 시험하고 기존모델 실제 read/외부 CSV read/source write/훈련 torch.load 차단을 확인했다. v4/v5 실패 산출물을 이어쓰지 않았다.

근거: [실패 receipt](RUN_TRAINING-failure.json), [v5 사전등록](preregistered-repair.md), 후속 [v6 실제 재생성 결과](../p2_clean_regeneration_20260905_v6/result.json). 이 실패영수증은 재생성 PASS가 아니다.

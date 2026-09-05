# C3 native 저장 경로 가드 정정 v6

v4와 v5는 각각 첫 seed 60epoch 학습/새 모델 저장 후 자기 산출물 SHA를 읽는 단계에서 막혔다. v5의 쓰기 이벤트 registry는 native `torch.save`가 Python open audit 이벤트를 거치지 않아 실패했다. 두 실행과 산출물/lock/봉인/실패영수증을 보존하며 재사용하지 않는다.

v6는 새 비어 있는 `03_model` 내부의 checksum 읽기를 허용하고 `torch.load`는 학습 단계 전면 금지한다. 새 출력 경로는 실행 전에 존재하면 중단하며, 모델 폴더가 빈 것을 명시 검증한 뒤 3 fresh fits를 단 한 번 실행한다. 모델 recipe/seed/epoch/weight/threshold/데이터/정책 변경 0이다.

학습 전 통합 합성 검사는 별도 프로세스에서 실제 native `torch.save`→자체 SHA→다음 seed 저장을 3개까지 수행한다. 같은 active audit hook 아래 기존 모델의 실제 read, 관계없는 CSV read, 원본 write 및 training `torch.load`가 거부되는지도 확인한다. 합성은 CPU의 작은 임의 Linear 모델이며 대회 fit 예산에 포함되는 데이터 학습이 아니다.

성공 이후 별도 PID inference와 또 다른 PID 전체26,061행 CSV replay/독립 QA를 수행한다. 공식키만 접근하고 sample값/hidden/old answer값/업로드는 0이다. guard 오류가 다시 발생하면 GPU 반복 없이 원인을 보고한다.

# C3 scratch 재생성 v5 — 가드만 기술 정정

v4 첫 seed scratch 학습 후 새 `.pt` SHA 읽기에서 가드가 중단한 실패는 보존한다. v5는 **새 비어 있는 03_model에서 3 seeds를 모두 새로 학습하는 1회 실행**이다. v4 모델·lock을 이어받지 않는다. 아키텍처, seed, epochs, 데이터, augmentation, 가중치, 손실, 보정강도, threshold는 변경하지 않는다.

수정은 이 실행에서 write한 모델의 hash 읽기만 허용하고 `torch.load`를 training 단계에서 명시적으로 금지하는 것이다. 기존 모델·OOF·답안·공식 키는 training 입력이 아니다. P3 GPU 해제 후 P2가 단독 사용하며 예상은 기존 full3 66.188초에 환경 여유를 둔 약 2분이다.

순서: `RUN_TRAINING` → 프로세스 종료 → 별도 PID의 `RUN_INFERENCE`. 출력 모델은 `artifacts/p2_clean_regeneration_20260905_v5/03_model/`, 답안은 `05_answer/submission_p2_clean_C3.csv`다. 공식 입력은 inference 단계의 station/layer/time 키만 읽고 sample temp·hidden truth·old answer values는 읽지 않는다. 과거 적격 C 답안 SHA와만 비교한다. 로컬 답안 생성은 허가됐지만 업로드는 0이다.

v4 실패 영수증·v5 합성 테스트 PASS는 빈 모델 재학습 PASS가 아니다. v5 실제 학습과 별도 PID 추론, schema/key/order/finite/hash 검사 결과가 있어야 재생성 PASS로 기록한다.

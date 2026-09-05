# P2 / OCN-02 제출 선택

리더보드 답안 채점 페이지는 `https://oceanaidata.org/app/problems/6`이다.
선택 파일은 `05_answer/submission_p2_clean_C3.csv` 하나이다.
열은 `station,layer,time,temp`, 26,061행이며 sample 키 순서와 같아야 한다. temp는 ℃ 단위 유한 실수이다.
`replay_p2_clean_C3.csv`는 검산용 동일 답안이지 다른 후보가 아니다. `.pt`, notebook, ZIP을 답안 CSV 입력란에 선택하지 않는다.

제안 제목: `P2 clean C3 source-only portable reproduction`
한 줄 요약: `배포 관측만으로 scratch 학습한 3-seed 수직 DeepSet의 명목 보간 잔차 복원; 독립 모델·추론 재현 검증.`

제출 전 `04_logs/inference-result.json`, `replay-result.json`의 schema/keys/hash를 확인하고 당일 잔여 횟수·공지·현재 승인을 별도 확인한다. 이 패키지는 업로드를 실행하지 않는다. 기존 제출과 SHA가 같으면 새 개선 후보로 중복 업로드하지 않는다.

최종 모델 제출은 리더보드 답안 채점과 별도다. 최종 잠금 효과와 당시 UI의 첨부/크기 제한을 확인한 후 승인된 경우에만 `02_code`, `03_model`, `04_logs`의 재현 영수증, `06_docs`, config/requirements/실행문서를 동봉한다. `01_data` 안내문은 포함할 수 있지만 운영진 원본 CSV는 ZIP에 넣지 않는다. Git에는 모델·답안·원본 데이터를 올리지 않는다.

# P2 적격 C3 — 빈 모델 폴더에서 학습·답안 재생성

`training-result.json`/`result.json`/`independent-qa.json`이 실제 완료 근거다. 단순 모델 reload와 달리 이 작업은 새 빈 `artifacts/p2_clean_regeneration_20260905_v5/03_model/`에서 20260901/02/03 seeds, 각 60epoch를 전부 scratch 학습한다. 과거 답안/모델/OOF는 학습에 읽지 않는다. v4 실패 산출물/lock은 보존하며 이번 학습에 사용하지 않는다.

## 위치 및 순서

- 배포 원본: `P2_DATA_DIR/observations.csv` (`C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore`). SHA 고정, 수정하지 않음.
- 실행 코드: `scripts/run_p2_clean_regeneration_20260905_v5.py`. v4 adapter의 고정 pipeline 호출을 격리된 v5 경로로 redirect한다. 원 canonical source/config는 수정하지 않음.
- 학습 모델: `artifacts/p2_clean_regeneration_20260905_v5/03_model/model_seed*.pt`.
- 첫 답안: `artifacts/p2_clean_regeneration_20260905_v5/05_answer/submission_p2_clean_C3.csv`.
- 별도 PID 검산 답안: 같은 `05_answer/replay_p2_clean_C3.csv`. 추가 제출 후보가 아니라 동일 bytes 재생 증거다.

```powershell
$env:P2_DATA_DIR = 'C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore'
# 새 승인된 실행에서만. 현재 소비된 lock을 지우거나 명령을 재실행하지 않는다.
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v5.py seal
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v5.py RUN_TRAINING
# 위 프로세스 종료 후 별도 프로세스 두 개
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v5.py RUN_INFERENCE
.venv-p1\Scripts\python.exe scripts/qa_p2_clean_regeneration_20260905_v5.py
```

공식 추론은 index/sample의 `station,layer,time` 열만 읽고 sample 수치·baseline CSV·hidden label·과거답안값은 읽지 않는다. 제출 형식은 **P2/OCN-02**, `station,layer,time,temp`, 26,061행이다. 리더보드 CSV 입력란에 `03_model` 파일을 넣지 않는다. 이 단계의 업로드는 0이다.

기술 정정은 신규 생성 모델의 hash 읽기 허용뿐이며 `torch.load`는 학습 단계에서 여전히 금지한다. canonical `train`→새 `VerticalDeepSet` 초기화→배포 관측 학습→모델저장 순서다. `bin17_anchor`, router/GI 답안, Public 역산 alpha를 쓰지 않는다. 역사적 `alpha40` 이름 모듈의 OAS 함수는 C3 경로에서 쓰지 않으며 재생성에서 호출 차단한다.

이 checkout의 source/env 경로를 이용한 로컬 재생성 검증이다. 데이터·코드를 모두 동봉한 portable 최종 ZIP, 타 머신/인터넷 차단/운영진 하드웨어 6시간 검증까지 완료한 것은 아니다. v4 실패영수증 PASS·v5 합성검사 PASS·실제 빈 모델 재생성 PASS를 구분한다.

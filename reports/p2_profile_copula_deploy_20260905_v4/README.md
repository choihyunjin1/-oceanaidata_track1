# P2 고정 full profile copula — local 후보 생성 계약

이 폴더는 `p2_profile_copula_residual_20260905_v4`의 **주평가 개선/추가 7·14일 결측 악화** 후보를 같은 full 강도 1.0 그대로 로컬 답안으로 만드는 별도 단계다. half/routing/새 threshold 탐색은 없다. 통과/실행 상태는 존재하는 `training-result.json`, `result.json`, `replay.json`, `independent-qa.json`으로 판단한다. 존재하지 않는 단계는 완료가 아니다.

선행조건: `p2_clean_regeneration_20260905_v6`가 비어 있는 `03_model`에서 C3를 scratch 학습하고, 새 PID 추론·전체 CSV 재생·적격 C 답안 SHA 일치를 독립 QA한 뒤에만 봉인/학습한다. 과거 C 모델·답안 또는 historical OOF에서 full 잔차를 가져오지 않는다.

고정 순서:

1. 재생성 C3 모델 3개를 새 `artifacts/p2_profile_copula_deploy_20260905_v4/03_model`로 해시 동일하게 복사한다(신규 backbone fit 0).
2. 배포 observations의 기존 적격 166,268행에서 재생성 C3 예측을 계산하고 `truth−C3`의 train-only rank-CDF/Ledoit–Wolf 모델을 **1fit**한다. 배포 학습 residual이며 in-sample 한계는 동일하다.
3. 새 PID `RUN_INFERENCE`: 공식 index/sample의 station/layer/time만 읽고 `C3 + full_copula_residual` 답안을 `05_answer/submission_p2_profile_copula.csv`에 만든다. sample temp/hidden truth/기존 답안값 읽기0.
4. 또 다른 PID `RUN_REPLAY`로 원본 공개 context부터 전체 26,061행을 다시 추론하고 CSV bytes SHA exact를 대조한다. `control_C3.csv`는 기준 해시 확인용이며 후보로 제출하는 파일이 아니다.
5. 독립 QA까지 통과한 후보만 root에 전달한다. 업로드0, 최종 모델 잠금0, Git0. 기존 source/model/answer/lock은 삭제하지 않는다.

```powershell
$env:P2_DATA_DIR = 'C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore'
.venv-p1\Scripts\python.exe scripts/run_p2_profile_copula_deploy_20260905_v4.py seal
.venv-p1\Scripts\python.exe scripts/run_p2_profile_copula_deploy_20260905_v4.py RUN_TRAINING
.venv-p1\Scripts\python.exe scripts/run_p2_profile_copula_deploy_20260905_v4.py RUN_INFERENCE
.venv-p1\Scripts\python.exe scripts/run_p2_profile_copula_deploy_20260905_v4.py RUN_REPLAY
.venv-p1\Scripts\python.exe scripts/qa_p2_profile_copula_deploy_20260905_v4.py
```

이는 현재 checkout에서의 모델·답안 재생성 검증이다. 별도 서버·운영진 하드웨어·portable 최종 ZIP·오프라인 환경 재현까지 검증했다고 표현하지 않는다. 데이터/모델/답안/lock은 Git 제외다. 내부 ΔRMSE로 공식 점수나 최고점을 보장하지 않는다.

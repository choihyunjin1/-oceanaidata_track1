# 완료된 패키지에서 빈 학습 폴더 만들기

완료 패키지의 기존 모델과 lock을 지우지 않습니다. root에 동봉한 `build_package.py`는 code/config/docs만 새 폴더로 복사하고 `03_model`, `04_logs`, `05_answer`를 비워 둡니다. 기존 모델/답안은 복사하지 않습니다.

```powershell
python build_package.py --output '../P2_fresh_reproduction'
python -I -B ../P2_fresh_reproduction/02_code/run.py RUN_TRAINING --cpu-threads 1
python -I -B ../P2_fresh_reproduction/02_code/run.py RUN_INFERENCE --cpu-threads 1
python -I -B ../P2_fresh_reproduction/02_code/run.py REPLAY --cpu-threads 1
```

먼저 P2_DATA_DIR를 공식 배포 폴더로 지정하고 requirements의 설치 환경을 준비합니다. 이 패키지를 새로 실행하는 것은 계산 작업이며 공식 업로드나 최종 모델 잠금이 아닙니다.

이번 실제 검증의 학습·추론·재현은 68.391/11.812/12.047초였습니다. 답안 SHA는 `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`입니다. 같은 장비/환경에서 검증했으며 다른 환경의 byte identity를 보장하지 않습니다.

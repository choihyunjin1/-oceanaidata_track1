# P2 C3 빈 모델 폴더 재생성 — 실제 PASS

결론: **새 빈 `03_model`에서 3seed×60epoch를 scratch 학습 → 새 PID에서 26,061행 답안 생성 → 세 번째 PID 전체 CSV 재생**이 완료됐고, 기존 적격 C3와 **SHA까지 정확히 일치**했다. 기존 모델을 불러 추론만 한 결과가 아니다. [result.json](result.json), [27-check 독립 QA](independent-qa.json), [training-result.json](training-result.json)이 완료 근거다.

- 적격 기준 답안 SHA: `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`.
- 전체 scratch training 69.094초, 3 fits의 순수 fit 합계 54.203초. 원래 166,268행 + mass-preserving augmentation 51,354행, 가중치 합 166,268을 재확인했다. seed 20260901/02/03, 60epoch, 모델/손실/가중치/규칙 변경 0.
- 학습은 배포 `observations.csv`만 사용했고 기존 model/OOF/답안/공식키 읽기0. inference는 공개 index/sample의 station/layer/time만 읽으며 sample값/hidden/old answer값0. 업로드0.
- source SHA 전후 일치. 서로 다른 training/inference/replay PID 및 전체 keys/order/schema/finite/hash를 확인했다. 새 모델 3개는 여기에서 생성됐으며 옛 답안은 읽지 않고 알려진 SHA만 대조했다.
- 사전 합성2 tests PASS에는 실제 native `torch.save` 3회와 SHA 읽기, 실제 금지 read/write, training `torch.load` 거부가 포함된다. Ruff PASS.

## 실패 비용과 범위

v4/v5는 각각 첫 seed의 60epoch 학습/저장 후 **자체 모델 SHA 읽기 가드 오류**로 중단했다. v4 첫 fit 19.859초, v5 시간은 보존된 progress에 있다. 두 미완료 attempt의 모델/lock/봉인은 삭제하지 않았고 v6에서 재사용하지 않았다. 이번 재생성 관련 GPU fit 비용은 실패 attempt의 첫 seed 2개 + 성공 v6 3개 = **5개 실제 학습**이며, v6 성공 fit count만 3으로 따로 기록한다. 가드 결함을 모델 실패나 과학적 NO_GO로 치환하지 않는다.

## 실행 입구와 파일 선택

원본 위치는 `P2_DATA_DIR=C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore`; 원본을 수정하지 않는다. 코드 [v6 driver](../../scripts/run_p2_clean_regeneration_20260905_v6.py)는 고정 canonical 학습/추론 함수를 새 독립 출력 경로로 redirect하며 기존 source/config를 바꾸지 않는다.

```powershell
$env:P2_DATA_DIR = 'C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore'
# 신규 승인된 비어 있는 출력 계약에서만 실행. 이 완료 경로/lock으로 재실행 금지.
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v6.py seal
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v6.py RUN_TRAINING
.venv-p1\Scripts\python.exe scripts/run_p2_clean_regeneration_20260905_v6.py RUN_INFERENCE
.venv-p1\Scripts\python.exe scripts/qa_p2_clean_regeneration_20260905_v6.py
```

모델: `artifacts/p2_clean_regeneration_20260905_v6/03_model/model_seed*.pt`.

기준 답안: `artifacts/p2_clean_regeneration_20260905_v6/05_answer/submission_p2_clean_C3.csv` — **P2/OCN-02**, `station,layer,time,temp`, 26,061행. `replay_p2_clean_C3.csv`는 동일 답안의 검산용이며 별도 새 후보가 아니다. 모델 `.pt`를 리더보드 CSV 입력란에 올리지 않는다.

이것은 현재 checkout에서 source→model→answer를 검증한 결과다. portable 최종 코드·환경 ZIP, 모든 데이터/코드 동봉, 다른 머신/네트워크 차단/운영진 하드웨어 6시간 검증이 완료됐다는 뜻은 아니다. 금지 bin17/역산 alpha/과거 답안체인은 학습 및 추론 경로에 없고, 역사 `alpha40` 모듈에서 import된 OAS 함수는 호출 차단 상태였다. 가드의 통제된 Python 경로 감사는 OS 전체 접근 감사와 구분한다.

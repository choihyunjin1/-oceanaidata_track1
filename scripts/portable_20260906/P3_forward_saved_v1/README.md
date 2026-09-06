# P3 forward candidate — 장기 저장 모델 추론 패키지

이 ZIP은 이미 source-only 전체 재학습과 독립 QA를 마친 **동일 후보의 모델 3개를 재생**합니다. 빈 모델에서 학습하는 cold 패키지가 아닙니다. 원 cold 실행의 6시간 제한을 우회하거나 새 학습 성공을 주장하는 용도로 쓰지 않습니다. 공식 최종 모델 잠금·업로드도 수행하지 않습니다.

## 포함/제외

- `03_model`: 정확한 full single CatBoost, full multi CatBoost, full loss router 3개만.
- `02_code`: 고정 후보·선택 특징 순서·recipe·source manifest 및 추론 코드. 모델 혼합은 cold와 동일한 router/단기 equal blend/장기 persistence 0.2이며 임의 배합이 아닙니다.
- `06_docs`: cold 학습/QA/답안 재생 hash 계보와 train-only shrink 선택 출처.
- 배포 관측 원자료, historical 모델, training cache, OOF, 검증 probe, 기존 CSV, logs, attempt locks, credentials는 ZIP에 포함하지 않습니다. 실행 중 historical cache/OOF/probe도 필요하지 않습니다.

## 실행

정확한 Python/라이브러리는 `02_code/requirements.txt`를 따릅니다. 배포 데이터는 별도로 제공하고 원본을 수정하지 않습니다. 이 추론은 CPU 2 threads이며 GPU 학습을 수행하지 않습니다.

```powershell
$env:P3_DATA_DIR='배포받은 P3_wave_forecast 폴더의 절대 경로'
# 원 연구 repo 경로를 추가 차단하려면 지정합니다. portable 코드에 개인 경로를 저장하지 않습니다.
$env:P3_DENY_REPO='원 연구 저장소 절대 경로'
python -I 02_code/infer.py --preflight
# 공식 입력 접근이 승인된 후, 존재하지 않는 새 출력 폴더를 지정합니다.
python -I 02_code/infer.py --infer --official-approved --output-dir '새 출력 폴더의 절대 경로'
```

출력 폴더의 `submission.csv`가 문제 3 리더보드 답안입니다. schema는 `case_id,station,lead_h,hs_pred`, 200 사례×6 leads(3/6/9/12/18/24h)=1,200행이며 공개 index 순서·키·중복·유한값·0~30m 범위·LF 줄바꿈을 검사합니다. 공개 context/index의 SHA를 전후 대조하고, cold가 승인한 동일 입력과 답안 SHA에 정확히 일치해야 CSV를 씁니다. sample/hidden truth/이전 답안의 값은 읽지 않습니다.

매 추론은 새 worker process에서 실행하고 supervisor가 **600초** 후 미완료 worker를 종료합니다. `process-receipt.json`은 PID/시간/종료 상태, `answer-qa.json`은 실제 답안 SHA와 계보를 기록합니다. 장기 재생성도 매번 새 출력 폴더를 사용해 기존 CSV/receipt를 덮어쓰지 않습니다. 같은 저장 모델의 재생 일치와 처음부터 다시 학습했을 때의 동일성은 별개의 주장입니다.

Python 네트워크 접근 훅과 파일 allowlist/원 repo 차단을 사용하지만 OS 방화벽·인터넷 차단 심사 환경을 인증한 것은 아닙니다. 최초 새 ZIP 추출 후 실제 새 PID 1,200행 exact 재생 결과가 나오기 전에는 이 패키지의 실제 실행 검증을 PASS로 표시하지 않습니다. 라이브러리/OS 차이 때문에 hash가 다르면 임의 값 보정이나 허용치 확대 없이 실패로 보존합니다.

최종 모델 재현 심사에는 별도 source-only cold 학습 코드와 그 실측 전체 시간/QA도 필요합니다. 이 저장 모델 ZIP만으로 학습 과정이 포함되었다고 주장하면 안 됩니다.

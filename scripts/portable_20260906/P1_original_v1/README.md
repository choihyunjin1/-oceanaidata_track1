# P1 원형 모델 — 전체 학습·추론 패키지

원본 O(XGBoost 1개) + B(event/day-balanced LightGBM 3개) + MS-TCN e150 3개,
원형 station/layer router와 일반 GI spike 규칙입니다. 최근 b2f17 전역 합집합 모델이 아닙니다.

## 현재 보장 범위

- 원래 저장 모델 7개로 `57844e…` 답안 전체 SHA 재생은 별도 감사에서 PASS했습니다.
- 배포 원본에서 생성한 80개 특징은 과거 train/test 캐시와 값·파일 SHA까지 일치했습니다.
- 이 패키지의 빈 모델 폴더 재학습 완료 여부는 **실제 `terminal.json`**으로 확인합니다.
  소스 생성·테스트 통과만으로 재학습 PASS라고 부르지 않습니다.
- 약 2시간은 과거 실행시간에 따른 예상입니다. 전체 준비 후 학습→QA→추론에 6시간 제한을 둡니다.
  GPU는 단독 사용합니다. O/B CPU8, MS의 CPU 지원 thread2 및 CUDA0 bf16입니다.
- 원래 router는 로컬 OOF에서 선택된 고정 일반 정책입니다. 전체 셀 선택 알고리즘은
  복구되지 않았으며 이 패키지는 셀/HPO 재선택을 하지 않습니다. 이를 자동 재적합이라고 표기하지 않습니다.

## 폴더

| 경로 | 역할 |
|---|---|
| `01_data/` | 배포 원본을 지정하는 자리. 원본은 복제/동봉하지 않음 |
| `02_code/` | 독립 소스·고정 설정·전체 실행 진입점 |
| `03_model/tree/` | 이 실행에서 처음부터 학습한 O/B 및 TRAIN-only 재생 probe |
| `03_model/mstcn/` | 이 실행에서 처음부터 학습한 3-seed MS, encoder, 소유 코드·검증 |
| `04_logs/` | 단계별 로그·이 실행만의 중간 추론 |
| `05_answer/P1_submission.csv` | 이 실행 모델로 생성한 최종 답안 |
| `06_docs/` | 환경·source provenance·답안 QA 영수증 |

## 실행

검증 환경 버전은 `06_docs/environment.json`을 따릅니다. GPU용 PyTorch가 필요합니다.
코드의 네트워크 차단은 Python audit-hook 검사이지 OS 전체 차단망 입증은 아닙니다.
외부 관측/가중치는 없고, 학습 당시 과거 모델·OOF·캐시·답안은 입력이 아닙니다.

1. 새 폴더에 소스 패키지를 두고 `03_model`이 비어 있는지 확인합니다.
2. `P1_DATA_DIR`을 운영진 배포 P1 디렉터리로 설정합니다. 다른 데이터 파일을 지정하지 않습니다.
3. 검증된 환경에서 `python 02_code/run.py all --data <배포 P1 경로>`를 실행합니다.
   `RUN_ALL.ipynb`는 같은 명령의 Jupyter 입구입니다.
4. `terminal.json`이 `TRAIN_TO_ANSWER_COMPLETE`인지 확인하고, `historical_answer_exact`도 별도로 확인합니다.
5. 기존 `ATTEMPT_LOCK.json`이 있으면 재실행하지 않습니다. 모델/lock 삭제로 재시작하지 말고 실패 원인을 확인합니다.

훈련 단계는 배포 `train.csv`만 읽고, 별도 PID의 tree/MS numerical replay를 통과한 뒤
공식 test의 관측값에 추론합니다. 원형 offline 관측 특징(양측 시간 문맥, 당해 관측의 수심 요약 등)은
훈련/추론에서 같은 규칙으로 계산합니다. 이는 train-fit-depth 변형과 다른 **명시적 원형 복원 범위**입니다.
encoder·예측 모델은 TRAIN에서만 적합합니다. 운영진의 최종 규정 판단을 대신하지 않습니다.

## 제출 방법과 금지 사항

- 답안 채점: P1(OCN-01)의 답안 업로드에 `05_answer/P1_submission.csv` 한 파일을 사용합니다.
  열은 `station,year,layer,time,label`, label 정수0/1, 169,011행입니다.
- `terminal.json`과 답안 SHA/validator를 먼저 확인합니다. 과거 SHA와 다르면 과거 28.909341점을 승계하지 않습니다.
- 모델 최종 지정은 답안 채점과 별개입니다. 현재 UI 첨부 제한·잠금 효과·운영진 요건과 사용자 승인을 확인해야 합니다.
  이 패키지는 업로드·최종 잠금·commit·push를 하지 않습니다.
- 데이터 원본은 재배포하지 않습니다. 모델 제출에 필요한 가중치·코드·환경·README만 별도로 포장합니다.
- 검증용 TRAIN probe는 정답 CSV가 아니지만 제출에 불필요한 내부 산출물 포함 여부를 최종 점검해야 합니다.
- 인터넷 차단 환경 설치/wheelhouse, 다른 GPU에서의 결정론, 새 독립 OOF 품질 확인은 별도 미검증 항목입니다.

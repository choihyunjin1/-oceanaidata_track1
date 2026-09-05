# P2 / OCN-02 — 독립 C3 학습·추론 패키지

이 패키지는 운영진 배포 데이터만으로 새 모델 3개를 학습하고, 그 모델만 읽어 P2 답안을 생성한다. 과거 답안·가중치·리더보드 역산 계수·외부 자료·사전학습 가중치를 사용하지 않는다. 연구 저장소가 없어도 실행된다. 원본 데이터는 재배포 금지이므로 포함하지 않는다.

## 폴더와 입력

- `01_data/`: 데이터 위치 안내만. 운영진 원본은 별도 보관하고 `P2_DATA_DIR`로 지정한다.
- `02_code/`: 독립 학습·추론 코드. 저장소 경로/import 의존성 없음.
- `03_model/`: 학습 산출 `.pt` 3개 및 `MODEL_MANIFEST.json`. 처음 학습할 때 반드시 비어 있어야 한다.
- `04_logs/`: stage별 consumed lock, 진행률, 학습·추론·재현 영수증.
- `05_answer/`: 모델 추론 CSV. 과거 CSV 복사 없음.
- `06_docs/`: 출처·제출 방법·패키지 manifest.

`P2_DATA_DIR`에는 운영진의 `observations.csv`, `test_index.csv`, `sample_submission.csv`가 있어야 한다. `baseline_interp.csv`, `score.py`, 비배포 관측, hidden answer는 실행에서 읽지 않는다. 학습은 `observations.csv`만 사용하고, 추론은 index/sample의 키 3열만 읽는다. 원본 파일은 수정하지 않는다.

## 환경과 실행

검증 환경: Python 3.12.10, NumPy 2.3.5, pandas 3.0.1, PyTorch 2.13.0+cu130, threadpoolctl 3.6.0. CUDA 사용이 가능한 동일 환경에서 검증한다. CPU threads 기본 2, DataLoader workers 0. 새 environment에서는 `requirements.txt` 버전의 패키지와 대응 CUDA runtime을 **미리 준비된 로컬 wheel**로 설치한다. 코드에는 다운로드·네트워크 호출이 없으며 실행 중 연결을 차단한다. 다른 하드웨어/버전에서는 동일 SHA를 보장하지 않는다.

PowerShell에서:

```powershell
$env:P2_DATA_DIR = '운영진_P2_데이터폴더'
python -I -B 02_code/run.py RUN_TRAINING
python -I -B 02_code/run.py RUN_INFERENCE
python -I -B 02_code/run.py REPLAY
```

패키지 밖 임의 작업 폴더에서도 `python -I -B <패키지경로>/02_code/run.py ...`가 동작한다. 제공된 `RUN_*.ps1`은 `$PSScriptRoot`를 기준으로 실행하므로 현재 작업 폴더와 무관하다. 세 명령을 각각 별도 프로세스로 실행한다. 학습이 끝나지 않았거나 모델/hash가 변하면 추론을 허용하지 않는다. `REPLAY`는 세 번째 PID에서 전체 답안을 다시 만들고 첫 추론 SHA와 대조한다.

동시 작업 자원 제한 시 각 명령에 `--cpu-threads 1`을 지정할 수 있다. 기본은 2이며 실제 적용 threads를 학습 영수증에 기록한다. 최초 portable 검증은 다른 문제와의 CPU 예산 때문에 1 thread로 실행한다. 이는 직전 v6 학습 내부 CPU 설정과 같다.

학습 실패 또는 완료 후 같은 lock에서 다시 실행하지 않는다. 기존 모델·영수증을 삭제해 재시작하지 말고 새 패키지 폴더에서 별도 승인된 검증을 수행한다. 실패와 SHA 불일치는 영수증에 남기며 결과에 맞추어 보정하지 않는다.

## 고정 학습 방법과 시간

공개 1/5/6/7/8층의 5개 token(8특징)과 11개 context를 받는 VerticalDeepSet(4,865 parameters)의 정규화 보간 잔차를 학습한다. 명목수심 보간 + 모델 잔차를 온도 단위로 복원한다. 공개 실제/명목 수심, 온도/염분 가용성 및 DOY/hour/M2 시간이 입력이다. 대상 2/3/4층 온도·염분은 특징 생성 전 마스킹한다.

배포 기간의 유효 target 166,268행, 3/7/14일 공개 T5/S5 block augmentation 51,354행, 원본별 총 가중치 질량 166,268을 유지한다. layer×월×날짜 균형 가중치, normalized Huber + 공개 온도 gradient penalty 0.01, AdamW(lr0.001, weight_decay0.0001), batch4096, 60epochs, seeds20260901/02/03이다. 추가 모델 선택·threshold·보정 fitting 없음.

동일 레시피의 직전 clean regeneration v6는 GPU scratch 3 fits 전체 69.094초였다. 이는 시간 예산의 출처이며 이번 portable 실행 시간은 `04_logs/training-result.json`과 inference/replay 영수증의 실측을 따른다. 6시간 충족은 검증 장비의 실측으로만 주장하며, 운영진 장비·격리 환경에서의 검증 완료를 대신하지 않는다. 해당 장비의 미리 설치한 CUDA 환경이 필요하며 무단 CPU fallback은 하지 않는다.

검증 목표의 기존 적격 답안 SHA는 `46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071`이다. 이 해시는 사후 재현 비교용이며 학습·추론 함수는 해시를 읽거나 결과를 맞추지 않는다. 과거 공식 점수는 SHA 일치가 확인된 경우에만 같은 파일의 역사적 점수로 연결한다. 패키지 생성만으로 공식 제출 완료 또는 최고점 갱신을 주장하지 않는다.

정확한 파일 선택과 주의사항은 [제출 방법](06_docs/SUBMISSION.md)을 따른다.

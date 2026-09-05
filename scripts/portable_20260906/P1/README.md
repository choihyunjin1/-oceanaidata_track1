# P1 재생성 기준 모델 — portable 검사 후보

XGBoost O + event/day weighted LightGBM B, 기존 clean v5의 80개 특징과 CPU4 순서를 보존한다. 학습은 inner B/O 및 전체 O/B의 4 fits. 과거 답안·외부자료·사전학습·리더보드 역산 계수는 입력으로 사용하지 않는다. 정상 급변 행을 새로 삭제/감량하지 않는다. 기존 day/event 가중치는 그대로다.

## 실행

Python 3.11 이상, `02_code/requirements.txt`와 같은 의존 환경을 준비한다. 오프라인 환경에서는 사전에 준비한 wheel을 로컬 설치한다. 원본 배포 P1 데이터 폴더를 `P1_DATA_DIR`로 지정한다. 데이터는 재배포 금지이므로 이 패키지에 넣지 않는다.

```powershell
$env:P1_DATA_DIR = '배포_P1_데이터_폴더'
python 02_code/run.py self-test
python 02_code/run.py train
python 02_code/run.py infer
python 02_code/run.py verify
```

`03_model`과 `05_answer`가 빈 별도 복사본에서 시작한다. 기존 모델/답안이 있으면 삭제하지 않고 실패한다. 새 전체 재학습은 새 빈 복사본에서만 수행한다.

- `01_data`: 배포 데이터 참조 안내용. 데이터 미동봉.
- `02_code`: 원 저장소가 필요 없는 코드·고정 설정·환경·원천 함수 hash manifest.
- `03_model`: 이번 학습으로 만들어진 모델·encoder·통계·inner 선택 recipe.
- `04_logs`: 실행 stdout/stderr를 저장할 곳.
- `05_answer/P1_submission.csv`: 위 모델에서 생성한 169,011행 답안. `station,year,layer,time,label`.
- `06_docs`: 학습·추론·replay 영수증과 상수 출처.

답안은 OCN-01 `/app/problems/5`의 답안 채점에 선택한다. `모델 최종 제출하기`와는 다르다. 공식 모델 제출 전 README·코드·가중치·환경과 현재 포털 첨부 요건을 다시 확인한다. 이 패키지를 생성한 것만으로 운영진 재현 검증 통과/최종 잠금을 주장하지 않는다.

원래 채점된 v5 SHA는 `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`. 비교용 메타데이터이며 학습 선택에 쓰지 않는다. 실행 결과 SHA가 다르면 원 점수를 승계할 수 없다. full 재학습 결정론과 저장 모델 replay는 별도 검증한다. 새 실행 실측 시간은 `06_docs/train-result.json`, `inference-qa.json`을 확인한다. 전체 ≤6시간은 실제 시험으로 판단한다.

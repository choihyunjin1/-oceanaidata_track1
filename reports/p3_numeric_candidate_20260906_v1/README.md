# P3 numeric-lead 후보: 로컬 최종 학습과 저장 모델 재생

이 작업은 **numeric-lead 내부 개선 후보**를 배포용으로 완성한다. 기존 clean 기준 답안 또는 hmax 제거 후보와 별개이며, 세 후보를 섞지 않는다. 최종 결과와 후보 SHA는 이 폴더의 `report-source.md`를 기준으로 확인한다.

## 실행 인터페이스

`P3_DATA_DIR`에는 주최 측 P3 배포 폴더를 지정한다. 데이터는 복사·배포하지 않는다. 저장소 루트와 `.venv-p1` 환경에서 다음 stage를 각각 새 Python 프로세스로 실행한다.

```powershell
$env:P3_DATA_DIR = '사용자가 보유한 P3 배포 폴더'
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage preflight
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage check
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage fit
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage replay
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage qa
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage infer
.venv-p1\Scripts\python.exe scripts/run_p3_numeric_candidate_20260906_v1.py --stage verify-answer
```

이는 실행 당시의 exactly-once 절차 기록이며 **이미 소비된 경로에 재실행하라는 명령이 아니다**. stage 출력은 exclusive-create로 보존된다. 실패한 fit이나 사용된 lock을 삭제/우회하지 않는다. 첫 preflight의 metadata 형식 오류는 실제 학습/봉인 전 정정됐고 원 실패 receipt를 남겼다.

## 산출물 구분

`artifacts/p3_numeric_candidate_20260906_v1/` 아래:

- `01_data/`: 데이터 동봉 없음. `P3_DATA_DIR`로 별도 배포본을 지정한다.
- `02_code/`: 실제 로드된 project Python source의 정확한 snapshot. 환경/의존 패키지는 별도다.
- `03_model/`: 새 `single.cbm`, `multi.cbm`, `router.joblib` 3개.
- `04_logs/`: training receipt, feature allowlist, train-derived replay probe, 독립 QA, answer/replay QA. probe는 학습 자료 기반 저장 모델 수치 재생용이며 내부 일반화 시험지가 아니다.
- `05_answer/submission.csv`: 1,200행 로컬 후보. `case_id,station,lead_h,hs_pred` 형식, UTF-8/LF.
- `06_docs/`: 패키지 역할/한계 안내용.

## 재현 주장 범위

새 빈 **full 모델 폴더**에서 2 backbone + 1 final router를 학습했다는 검증과, 새 PID 저장 모델/CSV exact replay를 구분한다. 이 경로는 exact-hash training feature cache와 기존 검증 OOF를 요구한다. `02_code` snapshot만 옮겨 독립 cold 학습을 할 수 있는 최소 ZIP이 아니며, 전체 5-fold부터 두 번 scratch로 재생성한 증거도 아니다. 그런 검증이 필요한 최종 패키지는 별도 작업이다.

실행은 CPU2, GPU0 multi 학습만 사용한다. 저장 모델 predict는 CPU이다. 3,600초 타이머는 새 fit 프로세스의 상한이며 전체 source-to-answer 6시간 독립 재현 증명으로 확대하지 않는다. Python audit hook의 network deny는 물리적/OS 전체 오프라인 검증이 아니다.

공식 입력은 training QA 이후 anonymous `test_context.parquet`의 허용 특징 및 `test_index.csv`의 키만 읽는다. sample 정답값/hidden/external/공개 점수 역산 입력/업로드/Git 작업은 이 runner에 없다. 최종 업로드 여부는 root가 별도 공식 receipt로 기록한다.

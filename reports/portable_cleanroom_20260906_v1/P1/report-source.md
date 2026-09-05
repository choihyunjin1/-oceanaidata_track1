# P1 portable baseline — two independent local rebuilds PASS

2026-09-06. **빈 모델 폴더에서 두 번 독립 학습한 4개 모델의 파일 SHA와 최종 답안 SHA가 모두 일치했다.** 학습→추론은 각각 **194.765초 / 215.079초**, 답안은 **169,011행**, SHA `5971e145f1ac38b8ee3e34cfd302973ba7a64b8873db11c354d3331221fdb28a`이다. 이는 기존 재생성 v5 기준 답안의 재현이며, 새로운 성능 개선이나 새 공식 채점 결과가 아니다.

## 시험과 경계

- 원 저장소 밖 `C:/Users/cedis/AppData/Local/Temp/ocean_p1_portable_20260906_b` 및 `_c`에서 각각 03_model/05_answer를 비운 신규 생성 상태로 시작했다. 기존 산출물은 삭제하지 않았다.
- 각각 inner LightGBM B/XGBoost O → full O/B, CPU4, 총 **8 fresh fits**. inner 선택은 이번 train에서 다시 계산했으며 과거 threshold/policy를 강제로 적용하지 않았다.
- 원 저장소 접근 deny(배포 데이터와 기존 venv 예외), Python socket audit guard, 빈 PYTHONPATH. 실행 코드는 독립 `02_code`에만 있다. 숨은 정답·외부 관측·과거 모델/답안 입력·업로드 0.
- 학습 시 official/test/sample 접근 0. 추론에서는 허용된 official 입력 169,011행과 sample 키만 읽었다. 학습과 추론/replay의 PID를 각각 분리했다.
- 같은 기존 `.venv-p1` / Python 3.12.10을 사용했다. **새 PC·새 의존환경 설치·OS 차원의 인터넷 차단 시험은 하지 않았다.** 운영진 최종 승인/모델 잠금 PASS와 구분한다.

## QA

추출 코드와 원본의 합성 특징/규칙/가중치/선택 parity, 파일/네트워크 경계 focused pytest **5 PASS**, runtime self-test 3 PASS, 관련 Ruff PASS. 실제 산출물 독립 [QA v2](independent-qa-v2.json) **20/20 PASS**: 각 fit/model/source/CSV 해시, schema/unique/binary/finite, 별도 PID, 6시간 예산, 두 전체 재학습 동일성.

최초 [byte-level QA](independent-qa.json)는 recipe 파일의 `inner_training_pid` 차이 한 개로 REVIEW였다. 모델/답안/학습 선택 값은 이미 exact였다. v2는 **실행 PID만 제외**한 구조 비교이며 숫자 예측 허용오차를 늘리지 않았다. 최초 영수증을 보존했다. 더 이른 `_a` 패키지 생성은 UTF-8 README 인코딩 오류로 학습 전 종료(0 fit)했고 역시 보존했다.

적합값/상수 출처는 패키지 `06_docs/constant-lineage.md`, 코드 원천 hash는 `02_code/source-manifest.json`, 숫자 리터럴 목록은 QA v2에 있다. 리터럴 목록 자체는 출처 증명이 아니며 기존 고정 하이퍼파라미터와 학습 적합값을 구분한다. 별도 물리범위 패치·셀별 정책·MS-TCN·bracket은 포함하지 않는다.

다른 문제 담당 에이전트의 정적 교차검토도 수행했다. QA JSON의 경계 metadata는 20개 계산 check가 자동 증명한 값이 아니라 [실제 실행 명령 관찰](execution-observations.md)과 guard 합성검사에 근거한다. Python hook만으로 native-library I/O 전체나 OS 격리를 입증하지 않는다.

## 보존 위치와 사용

- `artifacts/portable_cleanroom_20260906_v1/P1/run_a/`: 첫 검증 패키지.
- `artifacts/portable_cleanroom_20260906_v1/P1/run_b/`: 독립 두 번째 검증 패키지.
- 각각 `02_code` 코드, `03_model` 새 학습 모델, `05_answer/P1_submission.csv` 답안, `06_docs` 영수증. 배포 원본은 미동봉이며 `P1_DATA_DIR`로 불변 참조한다.
- `run_a/06_docs/P1_PORTABLE_REPRODUCIBILITY.zip`: 6,467,772bytes, CRC/member SHA PASS. [ZIP receipt](archive-receipt.json). 원본 데이터/로그/캐시는 제외했고 코드·이번 모델·답안·영수증을 포함한다.
- `scripts/portable_20260906/P1/build_package.py --help`의 새 경로 생성 후 README의 `self-test → train → infer → verify`를 따른다. 기존 모델을 지우지 않고 재시험마다 새 빈 패키지를 만든다.

현재 학습량/특징 연구 후보와 이 기준 패키지는 별개다. 이 작업에서는 commit/push/upload/final lock을 하지 않았다.

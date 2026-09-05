# PACKAGING_SPEC — 최종 재현 패키지 v2 (`artifacts/official_final_submission_v2_20260907/`)

## 0. 목적
운영진 재현 검증(인터넷 차단, 6시간, 1항 상수 리터럴·4항 학습 산출물 제거 후 예측 재생성)을 통과하도록, 문제별로 "배포 데이터 → 학습 → 가중치 → 예측 → 답안"이 코드만으로 재생성되는 원자적 폴더를 만든다. 기존 `artifacts/official_final_submission_20260905/`와 `scripts/build_official_final_submission_20260905.py`는 **수정하지 않고** 참고만 한다(폴더 계약·contract·노트북 골격은 그대로 따른다).

## 1. 빌더 `scripts/ocean_v2/build_final_package_v2.py`
입력: `configs/ocean_v2/final_package_v2.json`(문제별 최종 후보 id, 실행 디렉터리, 데이터 dir, 답안 SHA). 동작:
1. `P?/01_data/organizer_dataset/`에 배포 파일 하드링크(실패 시 복사) + `INPUT_MANIFEST.json`(파일별 bytes·SHA). 업로드 ZIP에서는 제외.
2. `P?/02_train/train_model.py`(= `python -m ocean_v2.p? train --config ... --out ../03_model`) + `TRAIN.ipynb`(실행 셀 3개: 환경 출력 → train 호출 → MODEL_MANIFEST 검증) + `TRAINING_LINEAGE.md`.
3. `P?/03_model/`: `weights/`(부스터 파일), `fitted_params.json`/`derived_constants.json`, `MODEL_MANIFEST.json`(파일 SHA·bytes·seed·하이퍼·학습 행 수·소요시간·라이브러리 버전), `cv_report.md`, `audit_constants.json`.
4. `P?/04_predict/predict_submission.py`(= `python -m ocean_v2.p? predict --models ../03_model --out ../05_answer`) + `PREDICT.ipynb`.
5. `P?/05_answer/P?_submission.csv` + `receipt.json`(행 수·SHA·validator 결과·생성 시각). 반드시 04_predict를 실제 실행해 생성(복사 금지).
6. `P?/06_submission/FORM.json`(제목·한 줄 요약·저장소 URL·비고·답안 SHA) + `FORMAT.md`.
7. `P?/07_source/`: `src/ocean_v2/common`, `src/ocean_v2/p?`, `configs/ocean_v2/p?_<final>.json`, `pyproject.toml`, `requirements-ocean_v2.txt`(정확 버전 고정). 다른 문제 코드·기존 `src/p?_*` 모듈·외부자료 모듈은 포함하지 않는다(필요 함수는 이미 복사돼 있어야 함).
8. `RUN_TRAINING.ps1`, `RUN_INFERENCE.ps1`(`--clean-room` 옵션: 03_model 비우고 학습부터 재생성 후 05_answer SHA 대조), `README.md`, `contract.json`(candidate id, 답안 SHA, 입력 SHA, 소요시간, 허용오차, 결정론 조건).
9. `upload/`: 문제별 `P?_official_final_core.zip`(07_source+02/04/06+README+contract+RUN_*), `P?_model_weights.zip`(03_model; 50 MB 초과 시 45 MB 조각 `P?_weights.partNN.zip` + `REASSEMBLY_MANIFEST.json`, 기존 `reassemble_p1_upload_20260905.py` 로직 복사), `P?_answer.zip`(05_answer). 모든 파일 ≤ 50,000,000 bytes. `MASTER_MANIFEST.json`(전 파일 SHA, 상태 `LOCAL_READY_NOT_UPLOADED`).

## 2. README 필수 항목(문제별)
환경(OS, Python 3.12.10, 패키지 정확 버전, CPU 스레드 수), 데이터 배치 방법(`P?_DATA_DIR`), 실행 명령(학습·추론), 실측 소요시간(학습/추론, 6h 한도 대비), seed·결정론 조건과 타 머신 허용오차, 사용 데이터(배포 6파일 SHA)와 외부자료·사전학습 0 선언, 파라미터 유래 정책(모든 적합 상수는 fitted_params/derived_constants.json, 코드 리터럴은 README 물리 상수만; `audit_constants.json` 첨부), 답안 SHA와 업로드한 답안과의 관계, 모델 요약(특징군·모델·CV·후처리 3~5문단, 방법론 심사용).

## 3. 클린룸 검증 절차 (`scripts/ocean_v2/cleanroom_verify.ps1`)
1. 새 임시 폴더에 `upload/`의 ZIP만 복사·해제(원 저장소 접근 없음). 2. 새 venv(또는 `.venv-p1` 복제)에 `requirements-ocean_v2.txt` 설치(오프라인 캐시 허용). 3. `P?_DATA_DIR` 지정 → `RUN_TRAINING.ps1 --clean-room` → `RUN_INFERENCE.ps1`. 4. 생성 CSV SHA를 `contract.json`·업로드 답안 SHA와 대조(트리 모델은 byte-exact 기대; 불일치 시 max|Δ| 계산해 허용오차 내인지 기록). 5. 총 소요시간 ≤ 6h 확인, `CLEANROOM_RECEIPT.json` 저장. 6. `pytest tests/ocean_v2 -q`, `audit_constants` 통과 확인.

## 4. 사용자 전달물 (Codex가 마지막에 작성: `docs/ocean_v2_codex/USER_HANDOFF.md`)
문제별: 업로드할 답안 CSV 절대경로·행 수·SHA, 최종 모델 업로드 파일 집합(파일명·bytes·SHA), FORM 값(제목/요약/저장소 URL/비고), 기대 Public 범위(sanity), 클린룸 결과, 삭제 권고 제출 목록(`reports/claude_recon_20260905/00_SUMMARY.md` §7 참조), 클릭 순서(답안 업로드 완료 확인 → 삭제 → 모델 최종 제출). 코드는 절대 업로드를 수행하지 않는다.

# 세 문제 독립 재학습 소스

현재 선택과 실제 검증·공식 제출 파일은 [최종 패키지 안내](../docs/FINAL_RELEASE_20260907.md)를 먼저 읽으세요. 각 문제 안의 pinned README는 원 패키지 생성 당시 설명이므로 최신 점수·완료 상태는 최종 보고서가 우선합니다.

**P3 주의:** 이번 전체 재학습은 완료했지만 채점본과 660행이 다릅니다. 채점 당시 저장 모델 ZIP은 채점 SHA exact를 재생했습니다. 새 cold 결과에 기존 점수를 승계하거나 운영진 재학습 허용오차 충족으로 해석하지 마세요.

이 디렉터리에는 **학습 코드·고정 설정·환경 명세·노트북**이 있으며, 모델/배포자료/답안은 없습니다. `SOURCE_EXPORT_MANIFEST.json`은 로컬 source-only ZIP에서 가져온 원 파일 해시입니다. 임의로 pinned 파일을 수정하면 실행이 중단됩니다.

1. 이 폴더를 원 연구 저장소 밖의 새 작업 위치에 복사합니다. 학습 완료 폴더를 재사용하지 않습니다.
2. `python PREPARE_DIRECTORIES.py`로 Git이 보존하지 않는 빈 `03_model`·`05_answer` 등의 폴더를 만듭니다. 이 명령은 파일을 지우지 않습니다.
3. 문제별 수치 환경 명세와 공통 `requirements-notebooks.txt`에 맞는 Python/CUDA 환경을 준비하고 `P1_DATA_DIR`, `P2_DATA_DIR`, `P3_DATA_DIR`에 운영진 배포자료 경로를 지정합니다. 원자료를 Git에 복사하지 않습니다. 기존 환경 실측은 torch 2.13.0+cu130이며 CPU wheel로 조용히 바꾸지 않습니다.
4. 각 문제 폴더를 작업 디렉터리로 열고 아래 순서를 따릅니다. GPU 작업은 동시에 실행하지 않습니다.

| 문제 | 학습부터 답안까지 | 핵심 모델 |
|---|---|---|
| P1 | `RUN_ALL.ipynb` 또는 `python 02_code/run.py all --data <배포 P1 경로>` | O XGBoost + B LightGBM 3개 + MS-TCN 3개 |
| P2 | `TRAIN.ipynb` 다음 `PREDICT.ipynb` | C3 DeepSet L120 3-seed |
| P3 | `TRAIN.ipynb` 다음 `PREDICT.ipynb` | CatBoost numeric single/multi + TRAIN-OOF router |

출력은 각 문제 `03_model`과 `05_answer`에 생성됩니다. 같은 실행의 코드→모델→답안 QA 영수증과 전체 SHA를 확인하세요. 전체 재학습이 과거 SHA와 다르면 과거 점수를 승계하지 않습니다. 보존한 모델을 재생하려면 별도 로컬 `SAVED_MODELS.zip`을 사용합니다. 여기서 과거 답안·weights를 복사해 학습한 척하지 않습니다.

P1 원형 셀 선택 프로그램 미복구, GPU/타 환경 byte 결정론과 새 venv·OS 차단망 미검증 등 제한도 최종 안내에 명시했습니다. 이 소스 공개는 운영진 최종 승인이나 모델 최종 지정이 아닙니다.

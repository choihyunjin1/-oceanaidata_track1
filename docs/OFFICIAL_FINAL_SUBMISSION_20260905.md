# 공식 최종 제출 로컬 패키지 — 2026-09-05

## 결론

각 문제를 `P1/`, `P2/`, `P3/`의 독립 실행 단위로 만들고, 내부를 운영진 데이터 → 학습 코드 → 저장 모델 → 모델 추론 답안 → 제출 양식 순서로 분리한다. 빌더가 과거 답안 CSV를 최종 예측 입력으로 복사하는 방식은 금지하며, `05_answer`는 반드시 `03_model` 가중치를 실제로 읽은 `04_predict`가 생성한다.

P1과 P3는 규정 준수 계보 중 확인된 역사적 최고점 답안을 저장 모델 추론으로 byte-exact 재현한다. P2는 역사적 최고점 실행 당시 가중치를 저장하지 않았으므로 같은 v52 레시피를 세 seed로 새로 학습하고 그 모델 출력의 새 SHA를 고정한다. 따라서 P2의 `0.424019 C / 28.012945점`은 역사적 후보 근거이며, 새 model replay가 같은 공식 점수라고 주장하지 않는다.

로컬 완성본은 Git 비추적 영역인 `artifacts/official_final_submission_20260905/`에 생성한다. Git에는 빌더, 학습·추론 코드, notebook template, 계약, 테스트와 이 문서만 보존한다.

실제 포털에서 어떤 파일을 어느 문제에 선택하고 어떤 폼 값을 입력할지는 [공식 제출 실행서](OFFICIAL_SUBMISSION_RUNBOOK_20260905.md)를 따른다. 다음 AI의 정찰 순서와 금지 경계는 루트 [AI_HANDOFF.md](../AI_HANDOFF.md), 기계 판독용 선택은 `configs/final_submission_portal_20260905.json`에 고정한다.

## 선택 계보와 재현 수준

| 문제 | 선택 계보 | 역사적 public 지표 / 점수 | 역사적 SHA-256 | 현재 모델 재현 |
|---|---|---:|---|---|
| P1 | `P1_1_E150_PLUS_GI_SPIKE2` | F1 `0.833548` / `28.909341` | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` | byte-exact |
| P2 | `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020` | RMSE `0.424019 C` / `28.012945` | `331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9` | 동일 레시피 fresh 3-fit, 새 SHA |
| P3 | `P3_REFINED_PUBLIC_OPTIMUM_20260827` | RMSE `0.583892 m` / `24.066168` | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` | byte-exact |

P3의 더 높은 과거 공식 점수는 KMA/ERA5 외부자료 계보라 제외한다. 세 패키지 모두 운영진 배포 데이터만 사용하며 관측자료 기반 사전학습 가중치는 사용하지 않는다.

## 문제별 원자적 폴더 계약

각 `P?/` 아래 구조는 동일하다.

```text
01_data/
  organizer_dataset/     # 로컬 전용 운영진 배포 파일, Git/업로드 제외
  INPUT_MANIFEST.json    # 파일명·크기·SHA-256
02_train/
  train_model.py         # scratch 학습 진입점
  TRAIN.ipynb            # 학습 과정과 모델 manifest 검증
03_model/
  weights/               # train code가 생산한 저장 모델
  MODEL_MANIFEST.json    # 데이터·seed·fit·trainer·weight 계보
  training_provenance/   # 역사적 학습 receipt/manifest
04_predict/
  predict_submission.py  # 저장 모델을 실제 로드하는 추론
  PREDICT.ipynb
05_answer/
  P?_submission.csv      # 모델 추론 산출물
  receipt.json
06_submission/
  FORM.json              # 홈페이지 입력값과 답안 해시
  FORMAT.md              # 정확한 CSV 양식
07_source/               # 사용된 모듈·역사적 exact trainer/config
RUN_TRAINING.ps1
RUN_INFERENCE.ps1
contract.json
README.md
```

패키지가 빠르게 만들어지는 이유는 P1/P3의 장시간 scratch 학습을 다시 수행하지 않고 과거 학습 당시 저장된 가중치와 provenance를 검증해 복사하기 때문이다. 결과 CSV를 복사하는 것은 아니다. P2는 해당 가중치가 없어 빌드 시 실제로 세 모델을 다시 학습한다. `RUN_TRAINING.ps1` 또는 `02_train/TRAIN.ipynb`의 명시적 toggle로 각 문제의 scratch 재학습 경로도 실행할 수 있으며 기존 certified weights를 덮어쓰지 않는다.

## 제출 CSV 양식

- P1: `station, year, layer, time, label`, 정확히 `169011`행, `label`은 정수 `0/1`.
- P2: `station, layer, time, temp`, 정확히 `26061`행, `temp`는 finite 실수(C).
- P3: `case_id, station, lead_h, hs_pred`, 정확히 `1200`행, `lead_h`는 `3/6/9/12/18/24`, `hs_pred`는 finite 실수(m).

모든 키와 행 순서는 해당 문제의 `sample_submission.csv`와 같아야 하고 pandas index 열을 쓰지 않는다. 실제 홈페이지 제목, 한 줄 요약, 저장소 URL, 제출 대상 답안 SHA는 각 `06_submission/FORM.json`을 단일 근거로 사용한다.

## 재빌드와 안전 경계

`scripts/create_final_submission_notebooks_20260905.py`로 notebook template을 생성한 뒤 `scripts/build_official_final_submission_20260905.py`에 세 배포 데이터 폴더와 검증된 모델 계보 경로를 전달한다. `--execute-notebooks`는 각 TRAIN notebook을 manifest-audit 모드로, 각 PREDICT notebook을 실제 모델 추론 모드로 별도 kernel에서 실행한다.

원본 배포 데이터, 답안 CSV, 대형 가중치와 파생 cache는 Git에 커밋하지 않는다. 업로드 묶음에서도 원본 배포 데이터를 제외하고, 50 MB가 넘는 P1 자산은 SHA가 있는 조각으로 분할한다. P1/P2의 `07_source/src`는 실제 학습·추론 진입점의 dependency-closed allowlist만 포함하며, 과거 외부자료 연동 모듈은 감사용 Git 이력에만 남고 최종 ZIP에는 들어가지 않는다. P1 조각은 core ZIP에 포함된 `REASSEMBLE_UPLOAD.py`로 part/source SHA를 검증하며 복원한다. 네트워크 업로드는 이 빌더가 수행하지 않는다.

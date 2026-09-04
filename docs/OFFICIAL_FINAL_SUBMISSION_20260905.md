# 공식 최종 제출 준비 — 2026-09-05

## 결론

P1, P2, P3는 서로 독립된 폴더와 Jupyter notebook으로 패키징한다. 각 notebook은 해당 문제의 운영진 배포 입력 해시를 확인한 뒤 정확한 frozen clean-lineage 후보를 만들고 schema, 행 수, 키 순서, finite/domain 규칙, SHA-256을 fail-closed로 검증한다. 한 문제의 실행은 다른 문제의 경로나 산출물을 읽거나 수정하지 않는다.

로컬 산출 위치는 Git 비추적 영역인 `artifacts/official_final_submission_20260905/`이다. Git에는 재현 코드, notebook template, 계약, 테스트와 이 문서만 보존한다.

## 선택한 clean-lineage 후보

| 문제 | 후보 | 공식 public 지표 | 문제 환산 점수 | 정확한 CSV SHA-256 |
|---|---|---:|---:|---|
| P1 | `P1_1_E150_PLUS_GI_SPIKE2` | F1 `0.833548` | `28.909341` | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` |
| P2 | `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020` | RMSE `0.424019 C` | `28.012945` | `331b1635bb036e773ff73487075e803b1308223e905c28b0d1494ea88b4d94c9` |
| P3 | `P3_REFINED_PUBLIC_OPTIMUM_20260827` | RMSE `0.583892 m` | `24.066168` | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` |

P3의 더 높은 과거 공식 점수는 KMA/ERA5 외부자료 계보이므로 최종 재현 패키지에서 제외한다. 세 후보 모두 운영진 배포 데이터만 사용하고 사전학습 가중치를 사용하지 않은 scratch 계보다.

## 원자적 디렉터리 계약

각 `P?/` 폴더는 다음을 자체 포함한다.

- `P?_final_submission.ipynb`와 실행 완료본
- `run_submission.py`, `common.py`, `contract.json`
- 정확 재현에 필요한 package-local frozen inference/model assets
- `outputs/P?_submission.csv`, `outputs/receipt.json`
- 학습·추론 계보 검토용 `source_audit/`
- 홈페이지 제목과 한 줄 요약이 적힌 `README.md`

원본 배포 데이터는 재배포 금지이므로 포함하지 않는다. 실행자는 `P1_DATA_DIR`, `P2_DATA_DIR`, `P3_DATA_DIR` 중 해당 문제 하나만 지정한다.

## 알려진 재현 경계

- P1 exact mode는 scratch 3-seed MS-TCN e150의 frozen output에 등록된 GI spike 2행을 add-only로 적용한다. 약 631 MB인 선택적 checkpoint 세 개는 45 MB 조각으로 나누며, reassembly script와 전체/조각 SHA를 제공한다.
- P2 원 실행은 scratch 3-fit을 약 74초에 완료했지만 checkpoint를 저장하지 않았다. 따라서 exact mode는 당시 배포된 ensemble output을 frozen asset으로 사용한다. 전체 v52 학습/materialization source와 근거는 함께 제공하며 이 한계를 숨기지 않는다.
- P3 axis component는 동봉 checkpoint에서 byte-exact 재현된다. historical original component의 현재 saved-weight replay는 최대 `0.0048767 m` 차이가 있어, 정확한 deployed original component와 두 clean scratch-model bundle을 함께 제공한다.

## 홈페이지에서 확인한 형식

2026-09-05 로그인된 모델 최종 제출 modal에서 제목은 필수이고, 한 줄 요약, 복수 결과 파일, 저장소 URL, 결과물 URL, 메모를 입력할 수 있다. 개별 파일 한도는 50 MB이다. 로컬 빌더는 모든 `upload/` 파일에 이 상한을 강제한다. 실제 최종 제출 버튼은 추가 답안 업로드를 잠그므로 이 준비 작업에서는 누르지 않는다.

## 재빌드

`scripts/create_final_submission_notebooks_20260905.py`로 notebook template을 생성한 뒤 `scripts/build_official_final_submission_20260905.py`에 세 배포 데이터 폴더와 frozen component 경로를 명시한다. `--execute-notebooks`를 켜면 각각 새 Jupyter kernel에서 위에서 아래로 실행하고 bounded receipt만 남긴다.

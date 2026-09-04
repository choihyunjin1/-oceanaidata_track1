# 공식 최종 제출 원자적 패키지 QA

## 결론

P1/P2/P3를 각각 독립 실행 가능한 폴더로 구성했고, 모든 답안은 `03_model`의 실제 가중치를 `04_predict`가 로드해 새로 생성한다. 과거 최고점 CSV를 예측 입력으로 복사하지 않는다.

- P1과 P3는 운영진 배포 데이터만으로 scratch 학습했던 저장 모델을 사용해 규정 준수 계보의 역사적 최고점 답안을 byte-exact 재현했다.
- P2는 역사적 v52 최고점 실행이 가중치를 남기지 않았으므로 같은 레시피를 3 seed로 새로 학습했다. 새 모델 답안은 재현 가능하지만 역사적 최고점 CSV와는 다르며, 역사적 공식 점수는 참고 근거일 뿐 현재 답안의 점수로 주장하지 않는다.
- 외부 관측·재분석·예보 자료, 관측 기반 사전학습 가중치, hidden truth를 사용하지 않았다.

## 선택 계보와 현재 재현 수준

| 문제 | 선택 계보 | 역사적 public 지표 / 점수 | 현재 모델 답안 SHA-256 | 역사적 답안과 일치 |
|---|---|---:|---|---|
| P1 | `P1_1_E150_PLUS_GI_SPIKE2` | F1 `0.833548` / `28.909341` | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` | 예 |
| P2 | `P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020` | RMSE `0.424019 C` / `28.012945` | `64f59fe7b28b3d60189e6fdef8ed00708da2f3c8066509f52395395d5a93f1ce` | 아니오 — 동일 레시피 fresh 3-fit |
| P3 | `P3_REFINED_PUBLIC_OPTIMUM_20260827` | RMSE `0.583892 m` / `24.066168` | `ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa` | 예 |

P3의 더 높은 과거 점수 계보는 KMA/ERA5 외부자료를 사용했으므로 최종 패키지에서 제외했다.

## 문제별 인과 체인

각 `P?/` 폴더는 다음 순서를 강제한다.

```text
01_data/       운영진 배포 데이터와 입력 SHA-256 (로컬 전용)
02_train/      scratch 학습 코드와 TRAIN.ipynb
03_model/      학습 산출 가중치·모델 manifest·학습 provenance
04_predict/    저장 모델을 실제 로드하는 추론 코드와 PREDICT.ipynb
05_answer/     모델 추론으로 생성한 제출 CSV와 receipt
06_submission/ 문제별 CSV 계약과 홈페이지 입력 양식
07_source/     정확한 역사적 trainer/config/모듈과 규정 문서
```

`RUN_TRAINING.ps1`은 재학습 진입점이고 `RUN_INFERENCE.ps1`은 저장 모델 기반 답안 생성 진입점이다. TRAIN notebook은 기본적으로 이미 완료된 장시간 학습 산출물의 해시와 provenance를 검증하며, 명시적 toggle을 켜면 별도 출력 폴더에 scratch 재학습한다. certified weight를 자동 덮어쓰지 않는다.

P1은 3 seed × 150 epoch MS-TCN 학습 코드를 포함한다. P2는 package build 중 동일 v52 레시피 3 fit을 실제 수행해 가중치를 저장했다. P3는 두 CatBoost/router 계보를 만든 역사적 trainer/config와 provenance를 포함하며, 간편 재학습 진입점은 배포 데이터에서 base model을 별도 폴더에 다시 학습한다. P3 최고점 exact 추론은 검증된 역사적 두 scratch-training chain의 저장 가중치를 사용한다.

## 제출 CSV 계약

- P1: 열 순서 `station,year,layer,time,label`, 정확히 `169011`행, `label`은 정수 `0/1`.
- P2: 열 순서 `station,layer,time,temp`, 정확히 `26061`행, `temp`는 finite 실수(C).
- P3: 열 순서 `case_id,station,lead_h,hs_pred`, 정확히 `1200`행, `lead_h`는 `3/6/9/12/18/24`, `hs_pred`는 finite 실수(m).

각 `06_submission/FORM.json`은 제출 파일 상대경로, 현재 답안 SHA-256, 역사적 최고점 일치 여부, 제목, 한 줄 요약, 저장소 URL을 지정한다. 각 `FORMAT.md`는 열·dtype·행 수·키 순서 계약을 사람이 읽을 수 있게 반복한다.

## 실행 검증

- focused pytest: `5 passed`
- Ruff: `PASS`
- 독립 Jupyter 실행: TRAIN 3개와 PREDICT 3개, 총 6개 notebook, error output 0
- 모델 기반 답안: P1 `169011`행, P2 `26061`행, P3 `1200`행; 열 순서·키 순서·finite/domain guard 통과
- 역사적 최고점 SHA 일치: P1/P3 참, P2 거짓임을 manifest와 FORM에 명시
- upload 묶음: 26개 모두 `50,000,000` bytes 이하, ZIP integrity 통과
- upload ZIP 내 운영진 원자료, `LOCAL_DATA_PATH.txt`, `.env`, `.pem`, `.key`: 0개
- P1 checkpoint 3개: 분할 조각의 순서별 streaming reassembly SHA가 원본과 일치
- 실제 네트워크 업로드: 0회

운영진 원자료, 답안 CSV, 모델/checkpoint, 파생 cache는 Git에 커밋하지 않는다. Git에는 builder, 학습·추론 코드, notebook template, 설정, 테스트, 문서와 이 QA 근거만 보존한다.

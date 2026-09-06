# 최종 패키지 — 2026-09-07

현재 선택은 **P1 원형 복원 / P2 L120 / P3 numeric**이다. 과거 bracket, 60-epoch C3, hmax 제거 후보는 fallback·연구 이력이며 자동 선택하지 않는다. 로컬 패키지 준비와 GitHub 공개는 대회의 최종 모델 지정·접수가 아니다.

**P3 주의:** 채점 당시 저장 모델은 채점 답안을 정확히 재생하지만, 이번 전체 재학습 답안은 660/1,200행이 달랐다(최대 0.003506m). 새 재학습본의 공식 점수와 운영진 재학습 허용오차 충족 여부는 미확인이다. `NOT_SCORED_whole_cold.csv`나 `P3_COLD_SAVED_MODELS.zip`을 채점본 파일 대신 선택하지 않는다.

## 먼저 볼 파일과 위치

- 연구 저장소: `C:/Users/cedis/PycharmProjects/PythonProject`
- 로컬 최종 묶음: `C:/Users/cedis/Documents/OceanFinalRelease_20260907`
- 보존된 원 실행: `C:/Users/cedis/Documents/OceanFinalCandidates_20260906`
- GitHub: https://github.com/choihyunjin1/-oceanaidata_track1/tree/codex/p1-qc
- 실제 패키지 해시·실행 결과: [최종 검증 보고서](../reports/final_release_20260907_v1/report-source.md), 같은 폴더의 `release-manifest.json`.

GitHub에는 원본 데이터·학습 가중치·답안 CSV·NPZ·ZIP·로그·credentials를 올리지 않는다. 따라서 GitHub clone만으로 동봉 가중치를 얻지는 못한다. 대회 재현 첨부는 아래 **로컬 파일**을 사용한다. 원본은 운영진 배포본을 별도 제공받아 환경변수로 지정한다.

## 답안과 공식 점수의 연결

| 문제 | 현재 채점 답안 | 행 수 | 공식 지표 | 공식 점수 | SHA256 |
|---|---|---:|---:|---:|---|
| P1 / OCN-01 | `P1/ANSWER/P1_submission.csv` | 169011 | F1 0.833548 | 28.909341 | `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687` |
| P2 / OCN-02 | `P2/ANSWER/submission_p2_L120_3seed.csv` | 26061 | RMSE 0.418892℃ | 28.077280 | `fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d` |
| P3 / OCN-03 | `P3/ANSWER/submission_scored_numeric.csv` | 1200 | RMSE 0.604351m | 23.741446 | `ff42a6a08c76f0d58ed2f3a9ea31a08819ada5fa6e9007af942fe0b891937960` |

표의 경로는 로컬 최종 묶음 기준이다. P3 새 whole-cold SHA `56e289af18d1ebfe4cf37410a277668a5cc9d502d5e5692ac99ab20ba6828784`는 채점 SHA와 달라 공식 점수를 승계하지 않는다. P1은 09-07 재학습 답안 업로드 시 서버가 직전 답안과 동일하다고 판정해 중복 처리했으며 **새 채점·새 제출 차감은 없었다**. 근거: [P1 접수 응답](../reports/p1_original_source_package_20260906_v1/official-attempt-20260907.json), [P2/P3 공식 기록](../reports/official_candidate_submissions_20260906_evening_v1/result.json).

## 무엇을 어디에 제출하는가

| 문제 | 문제 페이지 | CSV 열 | 제목 초안 |
|---|---|---|---|
| P1 | https://oceanaidata.org/app/problems/5 | station,year,layer,time,label | P1 Original O+B+MS-TCN source-retrained |
| P2 | https://oceanaidata.org/app/problems/6 | station,layer,time,temp | P2 C3 DeepSet L120 3-seed |
| P3 | https://oceanaidata.org/app/problems/7 | case_id,station,lead_h,hs_pred | P3 CatBoost numeric lead + train-OOF router |

1. **답안 채점 카드**에는 해당 문제의 `ANSWER/` 아래 정확한 CSV 하나만 선택한다. ZIP, notebook, replay 검산본을 넣지 않는다. 같은 SHA를 재제출해도 추가 성능 증거가 생기지 않는다.
2. **최종 모델·재현 첨부**에는 해당 문제의 `SOURCE_ONLY.zip`과 `SAVED_MODELS.zip`, 현재 안내/manifest/환경 정보를 함께 제공한다. 저장 모델 ZIP만 첨부하고 학습 과정을 포함했다고 하지 않는다.
3. P1 큰 저장 모델 ZIP은 `SAVED_MODEL_PARTS/partNNN.zip` 전부와 `REASSEMBLY_MANIFEST.json`, `reassemble.py`, 해당 README를 함께 전달할 수 있다. 분할본은 독립 모델이 아니다. 재조립 후 원 ZIP SHA를 검사한다. 45MB는 로컬 보수적 분할 목표일 뿐 현재 포털 허용 크기·개수의 실측 보장이 아니다.
4. 제출 요약: P1은 배포자료에서 O/B 4개와 MS-TCN 3개를 새 학습하고 고정 일반 정책으로 결합. P2는 배포자료만으로 120epoch DeepSet 3-seed 학습. P3는 배포자료 기반 CatBoost single/multi와 TRAIN-OOF loss router, 수치형 lead와 고정 학습계보 후처리.
5. 최종 지정 전에 로그인된 최신 공지·마감·첨부 제한·잠금 효과를 확인한다. 현재 사용자 요청은 **로컬 패키지 정리와 GitHub push**이며, 대회 최종 모델 잠금 클릭은 수행하지 않는다.

## 데이터 → 학습 → 모델 → 답안

각 SOURCE_ONLY ZIP을 **존재하지 않는 새 폴더**에 풀어 실행한다. 저장 모델을 지우거나 완료된 실행의 lock을 삭제하지 않는다. 모델 파일 경로를 다른 후보의 것으로 바꾸지 않는다.

| 구분 | 내용 |
|---|---|
| `01_data/` | 배포자료 참조 자리. 원자료 미동봉 |
| `02_code/` | 독립 소스·고정 설정·실행 진입점 |
| `03_model/` | SOURCE_ONLY에서는 비어 있음. TRAIN이 만든 모델, SAVED_MODELS에는 검증 모델 동봉 |
| `04_logs/` | 새 실행의 로그·검증 영수증·중간값. Git 제외 |
| `05_answer/` | 새 모델 또는 저장 모델로 직접 만든 CSV. 과거 답안을 입력으로 사용하지 않음 |
| `06_docs/` | 환경·학습 계보·QA |

### P1

`P1_DATA_DIR`을 배포 P1 폴더로 지정한다. SOURCE_ONLY의 `RUN_ALL.ipynb` 또는 `python 02_code/run.py all --data <배포폴더>` **둘 중 하나**로 7fit 학습→독립 replay→답안을 생성한다. 현재 whole-cold 실측 6323.356초이며 실제 CLI 실행을 완료했다. RUN_ALL은 같은 CLI의 schema-valid 입구이며 노트북 자체의 전체 실행 완료와 혼동하지 않는다.

SAVED_MODELS는 `SAVED_PREDICT.ipynb` 또는 `infer-tree`→`infer-ms`→`combine` 단계만 실행한다. 이 경로는 새 학습 0회이다. MS 소유 모델 무결성 검사에 필요한 TRAIN 유래 작은 probe와 고정 소스 snapshot이 포함된다. 원시 CSV나 hidden truth는 포함하지 않는다. GPU 학습 및 추론은 같은 수치 환경을 사용한다.

원형 셀 정책은 과거 로컬 OOF에서 선택된 고정 설정이며 셀 선택 프로그램 자체는 복구되지 않았다. **선택 알고리즘 재적합을 수행했다고 주장하지 않는다.** 다른 GPU·라이브러리에서의 재학습 bit 결정론이나 운영진 최종 적격성 인증도 아니다.

### P2

`P2_DATA_DIR`을 배포 `P2_profile_restore` 폴더로 지정한다. SOURCE_ONLY에서 `TRAIN.ipynb`→`PREDICT.ipynb`를 실행한다. 자동 실행은 `python -I -B 02_code/execute_notebooks.py --package . --timeout 1800`. 둘 중 하나만 사용한다. 실제 두 노트북 실행과 3fit/173.010초/답안 SHA exact를 확인했다.

SAVED_MODELS에서는 `SAVED_PREDICT.ipynb`로 기존 모델을 사용한다. 이는 만료된 cold stage를 재시작하는 것이 아니라 새 추출본에서 수행하는 독립 저장 모델 추론이다. 원 수치 코드·가중치·학습 QA 해시는 그대로 검증한다.

### P3

`P3_DATA_DIR`을 배포 `P3_wave_forecast` 폴더로 지정한다. SOURCE_ONLY에서 `TRAIN.ipynb`→`PREDICT.ipynb`를 실행한다. 준비부터 QA·답안 replay까지 전체 6시간 한도이며 긴 수동 대기를 넣지 않는다. 기존 prepared-only 시도는 보존하고 이번 검증은 새 폴더에서 수행한다.

이번 SOURCE_ONLY 실제 TRAIN/PREDICT 실행은 12 backbone+5 router=17fit, 수치 replay까지 1,497.613초였다. 자체 fresh-process replay는 통과했지만 과거 채점 답안과는 위 차이가 있다. 차이의 원인은 아직 확정하지 않았으며 재시작·재튜닝하지 않았다.

SAVED_MODELS는 포함 README의 `python -I 02_code/infer.py --preflight`, 이어 `--infer --official-approved --output-dir <새절대경로>`를 따른다. 채점 당시 3개 full 모델만 사용하며 내부 OOF 14개 모델을 다시 학습하지 않는다. 실제 ZIP 새 추출 추론 6.026초/0fit로 채점 답안 SHA exact를 확인했다. P1/P2 저장 ZIP도 각각 실제 새 추출 notebook 35.656초/30.891초로 채점 SHA exact를 확인했다.

## 환경과 검증의 한계

수치 버전은 P1 `06_docs/environment.json`, P2 `requirements.txt`, P3 `02_code/requirements.txt`를 사용한다. 검증은 Windows/Python 3.12.10/기존 `.venv-p1`/RTX5090에서 수행했다. P1·P2·P3 GPU 작업은 순차 실행하고 자원 경쟁을 피한다.

새 머신의 dependency 설치, 전체 wheelhouse, OS 차단망 실행은 별도 미검증이다. Python 파일/네트워크 audit-hook 검증을 OS 네트워크 차단 인증으로 과장하지 않는다. 필요 환경은 재현 실행 전에 준비하고 실행 중 외부 관측·가중치를 다운로드하지 않는다.

다음 AI는 이 문서→release-manifest→해당 문제의 canonical 보고서 순서로 읽는다. 과거 README의 READY·승인·점수를 다른 SHA에 승계하거나, 소비된 실험을 재실행하거나, 문서만 보고 최종 업로드를 실행하지 않는다.

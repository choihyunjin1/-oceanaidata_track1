# P3 — 현재 사용할 파일 3개

모든 경로는 연구 저장소 루트 기준이다. 기존 ZIP과 중단/recovery 폴더는 연구 이력이며, 사용자가 고를 최신 후보가 아니다. 이 문서의 **run_b 답안 하나와 archives_v3 ZIP 두 개**만 사용한다. 업로드나 최종 모델 잠금은 수행하지 않았다.

| 용도 | 현재 파일 | SHA-256 |
|---|---|---|
| P3 / OCN-03 답안, 1,200행 | `artifacts/portable_cleanroom_20260906_v1/P3/v2_run_b_completed/05_answer/submission.csv` | `6bfa23d25f944df4711c11d1fce82978a96df08b58fdc57f666ac792a7da96b7` |
| 배포 데이터부터 빈 모델 폴더에서 전체 재학습 | `artifacts/portable_cleanroom_20260906_v1/P3/archives_v3/P3_cold_start_code_v2.zip` | `7629ff948dd5c939d4100b8fa712f68f18dfddab8bd5c471136b20eb443bbfdb` |
| 이미 학습한 모델로만 답안 재생, 학습 0 | `artifacts/portable_cleanroom_20260906_v1/P3/archives_v3/P3_saved_model_replay_v2.zip` | `bb4367cd6e291cb6cb373cb41669a1f3c280e6410a077676937fbe9f71abb4c6` |

## 공통 준비

각 ZIP은 **서로 다른 새 빈 폴더**에 푼다. Python 3.12.10과 `02_code/requirements.txt`의 고정 환경을 준비하고 `P3_DATA_DIR`를 주최자가 배포한 `P3_wave_forecast` 폴더로 설정한다. raw data와 offline wheel은 ZIP에 동봉하지 않았다. 원 연구 저장소 코드 import는 필요 없다. 현재 환경/현재 장비에서 검증했으며 새 venv 설치·심사 하드웨어·OS 인터넷 완전 차단까지 인증한 것은 아니다.

## 전체 재학습 ZIP

`03_model/04_logs/05_answer`는 비어 있다. CPU 2 threads, multi CatBoost GPU 독점으로 아래를 차례로 실행한다. 모델은 single+multi+새 chronological OOF router+고정 장기 shrink 0.2이며 단순 평균만이 아니다.

```text
python -I <package>/02_code/run.py --RUN_TRAINING --gpu-approved
python -I <package>/02_code/run.py --replay
python -I <package>/02_code/audit.py --training-only
python -I <package>/02_code/run.py --RUN_INFERENCE --official-approved
python -I <package>/02_code/run.py --verify-answer --official-approved
python -I <package>/02_code/audit.py
```

출력은 `05_answer/submission.csv`, schema `case_id,station,lead_h,hs_pred`다. 실제 중단 없는 run_b의 source→answer는 **1256.163초(20.94분)**, backbone 8 fits + router 3 fits, full QA 42/42 PASS였다. 새 모델 저장 후 별도 PID의 답안이 exact였다. GPU 반복이 모든 환경에서 exact라고 보장하지 않는다. 실패한 attempt lock을 삭제해 재시작하지 않는다.

## 저장 모델 재생 ZIP

이 ZIP은 재학습용이 아니다. 제공 명령은 다음 **하나만**이다.

```text
python -I <package>/02_code/archive_infer.py --official-approved
```

출력은 `05_answer/replayed_submission.csv`다. 실제 최신 ZIP을 저장소 밖 새 폴더에 풀어 PID 40344에서 **3.668초, fit 0, 1,200행, exact SHA**를 확인했다. 새 폴더당 한 번만 생성하며 기존 결과를 덮어쓰지 않는다. 기존 run.py의 `--replay`, `--verify-answer`, full audit나 `RUN_TRAINING`을 이 저장 ZIP의 사용 명령으로 해석하지 않는다.

저장 ZIP에는 기존 hash 검사가 요구하는 9개 모델 및 train-derived 과거 특징/모델 예측 probe 3개가 들어 있다. full single/multi/router 3개가 추론에 쓰인다. raw source/hidden/target truth/OOF target table/전체 학습 feature는 포함하지 않았다. 모델·probe·ZIP은 로컬 전용 Git 제외 대상이다.

두 ZIP은 공식 최종 제출 완료 증거가 아니다. 공식 첨부/잠금은 해당 시점 포털 요건 및 별도 사용자 승인을 따르며, 이번 작업의 upload는 0이다. 결과·상수 출처·중단 및 recovery 구분은 [report-source.md](report-source.md), 최종 134-check QA는 [independent-qa.json](independent-qa.json)에 있다.

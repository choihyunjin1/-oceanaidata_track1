# P1 원형 복원 주장–근거 원장

외부 문헌이나 리더보드 역산은 이번 복원 근거로 사용하지 않았다. 아래 경로는 저장소 상대 경로다. 실제 입력/source/model의 전체 SHA는 `full-replay-result.json`에 있다.

| 주장 | 실제 사용 근거 | 한계 |
|---|---|---|
| 원본 O 가중치가 남아 있다 | `artifacts/runs/20260813T155254+0900_train_378a4e89/model_metadata.json`; 현재 `output/2026-08-20/retrain/P1/model.joblib`, SHA d4b60c… | 모델이 존재하고 receipt가 일치; 새 학습은 아님 |
| O는 당시 답안을 재현한다 | `artifacts/runs/20260813T155528+0900_reproduce_378a4e89/manifest.json`의 28243f…; 이번 saved/full 결과 동일 SHA | cached features 사용 |
| 실제 제출 B는 target-covariate 실험의 fallback 모델이다 | `artifacts/p1_target_covariate_density_ratio_xgb_v1/{preexecution_seal,manifest,result}.json`; `scripts/run_p1_target_covariate_density_ratio_xgb_v1.py::_fallback_candidate` | density ratio 주실험을 재실행하지 않음. fallback 자체는 event-day-balanced LGBM |
| B 가중치/답안 일치 | 위 models/P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.joblib SHA5f7933…; 당시 답안 decedb…; 이번 saved/full 결과 | 후속 full-deployment-fit 경로와 구분 |
| router는 로컬 OOF 근거가 있는 일반 조합 | `scripts/package_preregistered_submission_20260826.py::p1_candidates`; `artifacts/daily_submission_value_20260826_v1/report-source.md`; `scripts/build_p1_current_router_oof_anchor_v1.py` | 원래 셀 선택 실행 알고리즘 미복구. 같은 OOF 선택/평가 |
| router bit를 정확히 복원 | 이번 result.json, saved-replay-result.json 및 independent-qa.json | 새 모델 품질 검증 아님 |
| GI 2행은 원형에서 일반 규칙으로 생성 | `scripts/build_preregistered_3x3_evidence_round_20260827.py`; `scripts/build_deadline_probe_set_20260828.py::p1`의 novel & spike | 전체 파일에 있는 P2/P3 분기는 실행하지 않음 |
| 전체 28.9점 답안 재생 | 원형 `04_predict/predict_submission.py`의 MS 추론 연산을 재사용하되 router CSV/고정 patch 읽기는 제거; 이번 full-replay-result.json 최종57844e… | 저장 가중치/캐시 replay에 한정 |
| 과거 점수28.909341 | `artifacts/official_final_submission_20260905/P1/contract.json`의 후보 id/score/SHA, 이번 동일 SHA | 새 채점·실시간 리더보드 조회 아님 |
| 최근 새 후보는 원본과 다른 모델 | `reports/p1_champion_reconstruction_20260906_v1/tree-lineage.md`, `candidate-selection.md`; `reports/official_candidate_submissions_20260906_evening_v1/report-source.md` | 변화별 인과 기여도는 미분해 |
| 원래 패키지는 전체 scratch 연결 미완성 | `artifacts/official_final_submission_20260905/P1/02_train/train_model.py`, `04_predict/{p1_pipeline,predict_submission}.py`, contract의 retraining_entrypoint_scope | 원본 자체는 수정하지 않음 |

## 향후 에이전트가 반복하면 안 되는 오류

- b2f17 후보를 57844e 원본의 exact 복원이라고 부르지 않는다.
- router_anchor.csv라는 저장 형식만으로 원래 router 연산이 Public 역산이라고 단정하지 않는다.
- gi_spike2_patch.json의 고정 행 목록과, 그 목록을 만든 일반 model-disagreement spike 규칙을 구분한다.
- 특징/임계값/셀 조합을 바꾼 재학습 결과 불일치를 GPU 비결정 또는 원본 소실로 단정하지 않는다.
- saved replay PASS를 빈 모델 폴더 재학습/최종 운영진 검증 PASS로 표기하지 않는다.
- 원래 셀 선택의 전체 알고리즘이 미확인인 점을 숨기거나, 사후 작성한 선택 알고리즘을 역사적 원본이라고 부르지 않는다.

# 주장–출처 원장

열람일 2026-09-05. 로컬 기준 commit `535f94a1791f2398f82ad27659b58701513ab327`. 아래 로컬 연구 기록은 팀 내부 1차 산출물이며 독립 외부 논문과 구분한다. 원본 데이터 행은 열람하지 않았다.

| ID | 주장/용도 | 원출처·작성자·날짜 | 접근/한계 |
|---|---|---|---|
| O1 | 반환 점수 역산 파라미터 금지; 정상 모델 선택 참고 허용 | 운영본부, 「리더보드 반환 점수의 사용에 관하여」, 2026-09-02 10:41, [공지](https://oceanaidata.org/app/notices) | 로그인 Chrome 공지 modal 직접 열람. 개별 팀 제재 결과는 미확인 |
| O2 | synthetic-only 사전학습 예외 4조건, 6시간 | 운영본부, 「사전학습 가중치 관련 규정 보완」, 2026-09-01 12:26, [공지](https://oceanaidata.org/app/notices) | 직접 열람. CPU-only나 전면 pretrained 금지 아님 |
| O3 | 모델 코드/가중치 9월 7일 제출, 재현 검증 | 운영본부, 「데이터분석 대학부 결과물 제출·채점 방식 안내 (수정)」, 2026-08-12, [공지](https://oceanaidata.org/app/notices) | 직접 열람. 정확한 종료 시각/하드웨어 미확정 |
| O4 | 현재 우리 9위·81.125885, 문제별 최고/격차 | 운영본부, [리더보드](https://oceanaidata.org/app/leaderboard), 2026-09-05 18:17 KST 전후 snapshot | 직접 열람. 실시간 상태이며 모델 적격성 인증 아님 |
| O5 | 데이터 schema, 복원 가림, 사례/리드, 공개 P3 score anchors | 운영본부 배포 P1/P2/P3 `README.md`, 로컬 배포본 | README만 열람. 관측·test/sample/예측 CSV 값 미열람 |
| P11 | P1 package는 MS-TCN 학습 외 router CSV/GI patch 의존 | 팀, [predict_submission.py](../../scripts/final_submission_20260905/P1/predict_submission.py), [train_model.py](../../scripts/final_submission_20260905/P1/train_model.py), 2026-09-05 | 코드 직접 확인; 완전 재생성 실행은 미실시 |
| P12 | 기존 XGB/LGBM/router 학습 코드 재사용 가능 | 팀, [pipeline.py](../../src/p1_qc/pipeline.py), [learning curve runner](../../scripts/run_p1_meaningful_learning_curve_generation_v1.py), [router runner](../../scripts/run_p1_incumbent_preserving_mstcn_asrf_v2.py) | 담당 연구 lane 코드 확인; 통합 필요 |
| P13 | 장기 offset/drift 및 FP/FN 진단, oracle 한계 | 팀, [P1_FAILURE_RECON](../P1_FAILURE_RECON_2026-08-13.md), 2026-08-13; [anchor FN audit](../p1_anchor_false_negative_oracle_audit_20260831_v25/report-source.md), 2026-08-31 | root 직접 확인. 탐색 가설용이며 outer label gate 학습 금지 |
| P14 | TabPFN3 실제 실행·악화 | 팀, [P1 terminal result](../../artifacts/p1_tabpfn3_structural_transition_20260901_v1r1/result.json), 2026-09-01 | root 집계 직접 확인. 모델군 전체 불가능 주장은 아님 |
| P15 | trial18, DRO, depthmask, historical registry | 팀, [trial18 통합](../parallel_frozen_candidate_confirmation_20260830_v4/report-source.md), [DRO 통합](../parallel_robust_repair_cycle_20260829_v2/report-source.md), [depth](../P1_GORS_DEPTH_INVARIANCE_2026-08-13.md), [registry](../p1_v54_historical_promotable_candidate_registry_audit_20260901_v1/report-source.md) | 담당 lane 원장 대조. exact 설정의 실패로 한정 |
| P21 | P2 공식 v52 .424019℃/28.012945점, 직전 대비 +.012006점 | 팀, [official receipt](../p2_v52_official_submission_20260901_v1/official-submission-receipt.json), 2026-09-01 | 저장 영수증; 오늘 신규 replay 점수가 아님 |
| P22 | P2 prefix/reference 불일치, clip cap/raw 누락 | 팀, [v52 result](../p2_v52_score_priority_third_moment_input_gradient_20260901_v1/result.json), [v13 runner](../../scripts/run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py), [deployment result](../p2_v52_score_priority_deployment_20260901_v1/result.json) | root 핵심 집계 직접 확인, 담당 lane 학습/저장 코드 확인 |
| P23 | bin17 계보의 공식 점수 계수 및 패키지 의존 | 팀, [alpha50 config](../../configs/experiments/p2_seasonal_oas_alpha50_deploy_20260828.json), [final predict](../../scripts/final_submission_20260905/P2/predict_submission.py), [builder](../../scripts/build_official_final_submission_20260905.py) | root config/predict 직접 확인. 단순 상수 이동으로 해결 불가 |
| P31 | P3 외부자료 미사용 최고도 Public 적합 alpha 포함 | 팀, [public optimum builder](../../scripts/build_p3_refined_public_optimum_20260827.py), 2026-08-27; [reset report](../p3_clean_incumbent_reset_20260901_v1/report-source.md), 2026-09-01 | root 적합 코드 직접 확인. 외부자료 금지와 별개 문제 |
| P32 | clean 181episode/TabPFN25 악화·6h 부분 개선 | 팀, [P3 TabPFN terminal](../../artifacts/p3_tabpfn3_structural_transition_20260901_v1r1/terminal_result.json), 2026-09-01 | 담당 lane 확인. 순수 TabPFN 성능으로 해석 금지 |
| P33 | density는 0fit gate종료, KMA confirmation은 clean 아님 | 팀, [density metrics](../../artifacts/p3_target_mix_density_reweighted_catboost_v1/metrics.json), [confirmation runner](../../scripts/run_p3_catboost_confirmation_contract_repair_20260830_v3.py) | 담당 lane 코드/집계 확인; 계열 전체 NO_GO로 확대 금지 |
| P34 | 반복 노출과 fresh 사례 부족 | 팀, [fresh audit](../p3_lead_continuous_fresh_episode_confirmation_20260830_v3/result.json), 2026-08-30 | 담당 lane 확인. 재분할/dense 확대는 독립성 복원 아님 |
| S1 | 내부 검증의 선택 편향 일반 근거 | Gavin C. Cawley, Nicola L. C. Talbot, JMLR 11, 2010, [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html) | web 원문 소개 열람; 현재 팀의 원인에 대한 직접 실험 증거 아님 |
| S2 | leaderboard 반복 적응의 낙관성 일반 근거 | Avrim Blum, Moritz Hardt, ICML/PMLR 37, 2015, [The Ladder: A Reliable Leaderboard for Machine Learning Competitions](https://proceedings.mlr.press/v37/blum15.html) | web 원문 소개 열람; 특정 후보 손실량 예측 아님 |
| H1 | 결측/수심/기상 증강의 작은 probe 및 v2 제안 | 사용자 소유 `reports/claude_recon_20260905/`, `docs/ocean_v2_codex/`, 2026-09-05 | 참고 가설. 일부 상충/오래된 주장 발견; 원장 우선. 이번 데이터 재계산 없음 |
| S3 | 조건부 metric-to-point 크기 | 팀, [기존 headroom QA](../leaderboard_clean_headroom_research_20260901_v1/independent-qa.json), 2026-09-01; P3 배포 README | 경험식/공개 score policy 표시용. 공식 개선 예측·모델 계수 최적화에 사용하지 않음 |

## 수행한 표적 조사와 종료 판단

- 기억 registry에서 프로젝트 키워드 조회: 관련 항목 없음. 현재 저장소와 공지를 근거로 사용.
- 문제별 실험 원장/코드/결과 대조를 병렬 수행. raw rows, checkpoints, credential, 제출 답안 미열람.
- 최신 외부자료·점수 사용 규정은 web 검색만으로 원문 확인 불가하여 로그인된 공식 공지를 UI로 직접 열람.
- 기존 최고 기록과 실제 학습 경로, 가장 큰 내부평가 모순을 root에서 spot-check.
- 직접 논문 조회는 Cawley–Talbot 및 Ladder로 제한. 새 알고리즘 무차별 검색은 수행하지 않음.
- 남은 공백은 새 학습/집계로만 줄어드는 효과량과 환경 세부다. 계획 요청 범위를 넘어 실행하지 않고 종료.

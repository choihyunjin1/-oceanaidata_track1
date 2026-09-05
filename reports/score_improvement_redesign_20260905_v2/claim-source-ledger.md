# 주장–근거 원장

확인일 2026-09-05. 로컬 코드·집계 보고서가 팀 성과의 1차 근거이며, 웹 문헌은 방법론 정의를 보강한다. 행별 관측·정답·예측은 이 문서에 싣지 않는다. 오래된 보고서의 '공식 미제출'은 해당 작성 시점 범위이고 최신 채점은 L1을 따른다.

| ID | 주장/용도 | 직접 출처 | 신뢰 범위·한계 |
|---|---|---|---|
| L1 | 오늘 4개 후보 채점, 남은2/2/1 | [receipt](../official_score_repair_submissions_20260905_v1/receipt.json) | 직전 authenticated UI 기록을 현재 재열람. 이번 턴 portal 재조회 아님; 횟수는 실행 전 갱신 |
| L2 | P1 unseen-year nominal-depth missing | [feature_pair/stats_fit](../../scripts/run_p1_score_repair_20260905_v1.py), [fulltrain report](../p1_clean_control_fulltrain_20260905_v1/report-source.md) | root 코드 대조. 수심 파생 특징의 문제이지 raw depth 자체가 전부 없다는 뜻 아님. 점수 영향 미측정 |
| L3 | layer ordinal은 연도 간 동일 물리 수심 아님 | [data.py add_depth_regime](../../src/p1_qc/data.py), [features.py _depth_metadata](../../src/p1_qc/features.py) | station×layer 무조건 fallback 배제, current-depth 대조의 출발점 |
| L4 | 최고점 P1과 오늘 2-tree는 다른 구조 | [AI handoff](../../AI_HANDOFF.md), [P1 fulltrain report](../p1_clean_control_fulltrain_20260905_v1/report-source.md), [historical ledger](../historical_model_reaudit_20260831_v1/candidate-ledger.json) | 최고점 전체 적격 판정 아님. 부품 출처 감사 필요 |
| L5 | P1 decoder 정책 실패와 always-on 진단을 구분 | [decoder report](../p1_score_repair_decoder_20260905_v1/report-source.md) | −0.000883 vs +0.001617, 후자는 사후 진단. 과거 점수환산 문구는 이번 계획에서 재사용 안 함 |
| L6 | P2 normalized Huber/domain objective | [training_arrays/fit_model](../../scripts/run_p2_score_repair_20260905_v1.py), [scale](../../src/p2_restore/normalized_curvature_residual.py) | root 코드 대조. ℃ MSE가 반드시 더 좋은 것은 아님 |
| L7 | P2 seasonal gain와 pooled gain은 다름 | [result report](../p2_score_repair_20260905_v1/report-source.md), [config](../../configs/experiments/p2_score_repair_20260905_v1.json) | 69,850행,3fold. 가을1fold는 반복 노출된 historical 표면 |
| L8 | P2 relative feature와 물리문맥 대안 | [pipeline build_arrays](../../scripts/final_submission_20260905/P2/p2_pipeline.py), [P2 must-read](../../01_P2_MUST_READ_FIRST.md) | 관측 분포 재분석 없이 코드 계약만 확인. 신규 tree의 성능/시간 미측정 |
| L9 | P3 clean OOF exact 재현과 시간 | [deploy report](../p3_score_repair_deploy_20260905_v1/report-source.md), [runner](../../scripts/run_p3_score_repair_deploy_20260905_v1.py) | 원래 전체재현1678.413초, 새 가중치/보정 시간 보장 아님 |
| L10 | P3 log component-loss router | [ComponentLossRouter](../../src/p3_wave/loss_router.py), [component runner](../../scripts/run_p3_corrected_repeated_forward_catboost_v1.py) | root 직접 확인. 직접 SSE가 OOS 우수할지는 가설 |
| L11 | 기존 P3 보정 계획과 새 저차원 대조는 다름 | [prequential calibration config](../../configs/experiments/p3_corrected_prequential_calibration_v3.json) | 설정 파일은 실행 증명이 아님. corresponding runner/artifact 부재로 '이미 실패' 단정 금지 |
| L12 | 외부자료·Public 역산 계보 제외 | [organizer policy](../../00_ORGANIZER_DATA_POLICY.md), [P3 must-read](../../02_P3_MUST_READ_FIRST.md), [P2 must-read](../../01_P2_MUST_READ_FIRST.md) | 최상위 로컬 정책, 새 실행 전 최신 운영진 변경 확인 |
| W1 | 반복 모델선택의 편향과 노출 이력 관리 | [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | 방법론 논문. 우리 모델 상승폭·금지조항의 근거로 확대하지 않음 |
| W2 | 메타 학습에는 교차 예측/훈련 분리 필요 | [sklearn StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) | 공식 API 문서. 기본KFold를 우리 시계열에 자동 적용하지 않음 |
| W3 | 시간·그룹 의존에 맞는 분할 | [sklearn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | 공식 가이드. 우리 purge 길이는 팀 데이터 계약에서 옴 |
| W4 | 절대 squared error 정의 | [PyTorch MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html) | loss 수학만 사용. 설치 패키지 upgrade 지시 아님 |
| W5 | SmoothL1의 구간별 제곱/선형 손실 | [PyTorch SmoothL1Loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html) | implicit w/scale²는 이 정의와 로컬 target 식의 대수적 추론 |

외부 자료 조사에서 웹 관측·재분석·예보·정답 후보를 수집하지 않았다. 문헌에 실린 타 데이터 성능을 예상 공식점수로 전용하지 않았다. 원본 README 세 개는 과제 계약 확인에만 썼고 데이터셋을 재배포하지 않는다.

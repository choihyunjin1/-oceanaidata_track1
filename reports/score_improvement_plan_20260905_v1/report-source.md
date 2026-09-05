# 점수 개선 계획의 조사 결론

대상: P1/P2/P3 연구 사용자 및 후속 실행 AI. 작성 2026-09-05. 기준 commit `535f94a1791f2398f82ad27659b58701513ab327`.

## 결론

**현재 가장 타당한 전략은 새 모델 전면 도입이 아니라, 기존 자산을 올바른 비교·학습·재현 경로로 연결하고 점수의 실제 오차 원인에 작은 변경을 집중하는 것이다.** P1은 장기 이상 표현과 양방향 오분류 수정, P2는 배포와 일치하는 복원 검증 및 공개층 결측/수심, P3는 clean 예측 상보성과 기상 결측 증강이 우선이다. 아직 새 실험을 하지 않았으므로 실제 점수 상승을 확정할 수 없다.

후속 실행용 산출물: [점수 개선 실행 계획](../../docs/SCORE_IMPROVEMENT_PLAN_20260905.md).

## 조사 범위와 방식

현재 저장소의 정책·handoff·dataset README·실험 코드·집계 결과·제출 원장, 2026-09-05 Claude 정찰 문서를 대조했다. 문제별 세 연구 에이전트가 독립적으로 기존 실적과 코드 경로를 조사했고 root가 P1 패키지/TabPFN 결과, P2 comparator/anchor, P3 공식 점수 적합 코드를 직접 재확인했다. 외부자료는 방법론 논문과 운영진 공지에 한정했으며, 데이터 값·hidden truth·제출 CSV를 열거나 모델을 실행하지 않았다.

현재 새 Claude 문서는 사용자 소유 미추적 변경이며 설계 제안일 뿐 승인된 규정이나 실행 지시가 아니다. 이에 적힌 Public 예상치, 과거 미실행 주장, CPU-only 제한을 그대로 채택하지 않았다. 코드/데이터/기존 artifact는 수정하지 않았고 계획 문서만 추가했다.

## 핵심 근거

### 1. 현재 최고점과 최종 적격 후보는 다르다

9월 5일 18:17 KST 전후 로그인된 [공식 리더보드](https://oceanaidata.org/app/leaderboard)에서 우리 팀 9위, P1 28.909341/P2 28.012945/P3 24.203599, 합계 81.125885를 확인했다. 문제별 최고는 32.262806/28.674902/24.784043이다. 남은 점수 차이는 기회 단서일 뿐 획득 가능한 효과량의 증명은 아니다.

로그인 [운영진 공지](https://oceanaidata.org/app/notices)의 9월 2일 「리더보드 반환 점수의 사용에 관하여」는 반환 점수 역산으로 모델 계수/임계값/파라미터를 정하는 행위를 금지하면서 정상적 모델 선택의 참고는 허용한다. 9월 1일 공지는 synthetic-only pretrained 모델의 네 조건을 재확인한다. 8월 12일 수정 제출 안내는 9월 7일까지 실제 모델 코드/가중치 제출 및 재현 검증을 명시한다. 정확한 마감 시각/검증 하드웨어는 이번 공지 열람으로 확정되지 않았다.

- P1 최종 추론은 router CSV와 GI 키별 patch를 사용하지만 학습 코드는 MS-TCN을 담당한다. 상위 XGB/LGBM/router 코드는 있으므로 연결 복구 대상이지 전체 재작성 근거는 아니다. [추론 코드](../../scripts/final_submission_20260905/P1/predict_submission.py), [학습 코드](../../scripts/final_submission_20260905/P1/train_model.py).
- P2 v52는 `anchor + 0.2*clip(model-anchor, -2.5, 2.5)`이며 anchor 계보에 공식 점수에서 선택한 계수가 있다. [최종 추론](../../scripts/final_submission_20260905/P2/predict_submission.py), [OAS alpha 선택 기록](../../configs/experiments/p2_seasonal_oas_alpha50_deploy_20260828.json).
- P3 외부자료 미사용 최고 0.583892m/24.066168점은 12/18/24h에 Public RMSE로 적합한 quadratic optimum alpha를 적용한다. 원래 모델 0.607071m와 구분해야 한다. [적합 코드](../../scripts/build_p3_refined_public_optimum_20260827.py), [기록](../p3_clean_incumbent_reset_20260901_v1/report-source.md).

운영진의 실제 개별 제출 판정은 미확인이다. 이 조사는 자동 철회·삭제·실격 확률을 제안하지 않는다. 다만 확인된 금지 방식의 계수를 새 최종 후보로 계승하지 않는다.

### 2. P1: FP/FN을 모두 수정할 수 있는 작은 비교가 필요하다

과거 동일 OOF 진단에서 XGB FN의 92.7%는 offset/drift, FN의 86.1%는 확률 0.05 미만이었다. 이는 새 후보 선택값이 아니라 표현 개선 가설의 근거다. 단순 threshold 인하, 기존 양성 OR 보존만으로는 한계가 있다. [실패 정찰](../P1_FAILURE_RECON_2026-08-13.md).

후속 anchor 진단에서는 FN의 98.61%가 truth event interior였다. 이는 **모두 이미 탐지된 사건**이라는 뜻이 아니며 직접 gap 뒤의 FN은 0이었다. 따라서 단순 gap 복구/구간 늘리기를 해결책으로 단정하지 않는다. [후속 진단](../p1_anchor_false_negative_oracle_audit_20260831_v25/report-source.md).

TabPFN3는 이미 11 fits/11,169.57초 실행하여 Q2 F1 0.79227557→0.71942311로 악화했다. +1,098행 중 TP 추가는 46, FP 추가는 1,052였다. 라이선스 때문에 미실행이라는 새 문서는 오래된 상태다. [terminal 집계](../../artifacts/p1_tabpfn3_structural_transition_20260901_v1r1/result.json).

trial18, Group-DRO, depth masking, 단순 run extension의 구체적 실패는 원장에 남아 있다. 이를 무조건 재실행하지 않고, 기존 LGBM 표현 한 축과 anchor 수정 자유도를 동일 inner/outer 평가에서 비교한다.

### 3. P2: 내부 comparator 및 학습 시점 불일치가 가장 큰 문제다

v52 내부 결과 reference 3.08571462℃, candidate 3.03306301℃, Sep/Oct reference 4.88213505℃가 저장돼 있다. 이 reference를 이긴 +0.05℃ 규모를 공식 0.424019℃ 기준의 개선으로 직접 해석할 수 없다. prefix-only 내부 학습과 미래 포함 full-history 배포 차이는 실제 코드로 확인된다. [결과](../p2_v52_score_priority_third_moment_input_gradient_20260901_v1/result.json), [내부 러너](../../scripts/run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py), [배포 결과](../p2_v52_score_priority_deployment_20260901_v1/result.json).

최종 clip은 p95/p99/max가 모두 0.5℃에 닿으며 raw absolute model은 NPZ에 저장되지 않는다. clip된 예측에서 standalone 모델을 역복원할 수 없으므로 기존 기록만으로 raw 후보들의 전수 순위가 나왔다고 말할 수 없다. 새 비교에서는 raw per-seed 예측, 단독 RMSE, mask별 SSE를 남긴다.

후속 계획은 복원형 공통 평가에서 v23/v52/OAS를 비교하고, 연속 결측 증강과 목표 actual depth를 별도 ablation으로 측정한다. Claude 소규모 probe의 수치와 결측 비율은 이번 turn에서 데이터를 재계산한 값이 아니므로 재현 대상 가설로만 쓴다.

### 4. P3: 반복한 것과 아직 학습하지 않은 것을 구별한다

78h 분리와 hs≥1.5, 6리드 유효성을 갖춘 181사례 전방 평가는 이미 있다. first-eligible greedy는 로컬 모사이며 운영진의 실제 선택 절차라는 근거는 없다. 추가 dense anchor가 새로운 독립 사례를 만드는 것도 아니다. [데이터 함수](../../src/p3_wave/data.py), [fresh 감사](../p3_lead_continuous_fresh_episode_confirmation_20260830_v3/result.json).

TabPFN3의 18fit/251.5초 결과는 25% 혼합 0.78324918m 대 fallback 0.77910484m다. 전체 개선은 없지만 6h의 상보성은 사후 탐색 가설로 남길 수 있다. 이 수치와 순수 TabPFN 단독 성능을 혼동하지 않는다. [terminal](../../artifacts/p3_tabpfn3_structural_transition_20260901_v1r1/terminal_result.json).

density reweighting은 ESS gate에서 training receipts 없이 끝난 사례가 있으므로 학습 성능 실패로 전 계열을 닫지 않는다. KMA correction을 양쪽에 넣은 CatBoost confirmation은 과적합 경고로 보존하되 clean 개선의 증거로 쓰지 않는다. [density 집계](../../artifacts/p3_target_mix_density_reweighted_catboost_v1/metrics.json), [confirmation 코드](../../scripts/run_p3_catboost_confirmation_contract_repair_20260830_v3.py).

## 근거 공백 및 처리

| 질문 | 확인 수준 | 남은 공백/실행 조치 |
|---|---|---|
| 최고점 파일 전체를 원본부터 재생성하는가 | 패키지의 부분 의존 확인 | 상위 학습 부품 연결 및 새 통합 재현 필요 |
| P1 장기 표현/OR 해제가 개선하는가 | 구조적 한계와 과거 오류 분포 확인 | 현재 control 동일 fold 대조가 필요; 상승 미확정 |
| P2 실제 depth/공개층 결측이 주원인인가 | 평가 불일치는 코드로 확인, 새 probe는 별도 기록 | aggregate 재검증 및 한 축 대조 필요 |
| P3 기상 결측 증강이 개선하는가 | 가설/소규모 보고 존재, 상충 probe 존재 | 같은 control/seed/split 12fit 이내 대조 필요 |
| 새 공식 점수는 얼마인가 | 미확인 | 조건부 metric-to-point 크기만 표시 |
| 제출 마감·실행 환경 세부 | 공지상 9월 7일 및 6시간 제한 확인 | 정확한 시간, 하드웨어, 전체/문제별 범위 추가 확인 |

## 검증을 바꾸는 원칙

모델 선택의 과적합은 알고리즘 간 차이만큼 커질 수 있고, 동일 leaderboard에 대한 반복 적응도 낙관적 평가를 만든다. 이 일반적 근거는 현재 팀의 실패 원인을 단독 증명하지는 않지만, 동일 모집단·inner fitting·후보 수 제한·회고 평가의 정직한 명명을 지지한다. [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html), [Blum & Hardt, ICML 2015](https://proceedings.mlr.press/v37/blum15.html).

강제 차단은 출처/누출/키/재현에 적용하고, 효과량 최소 3점·모든 월 양수·anchor 삭제 0 같은 자체 기준은 위험 지표로 재분류한다. 그러나 gate 완화가 과거 음수 실험의 성능을 바꾸지는 않는다.

## 조사 종료 사유

문제별 코드/원장에서 재사용 대상, 기존 실패, 규정 계보, 다음 최소 비교를 확인했고 가장 중요한 규정은 운영진 원문으로 검증했다. 추가 광범위 신방법 검색보다 실제 공통 평가와 bounded ablation이 다음 의사결정을 바꿀 가능성이 높아 여기서 계획을 확정한다. 새 데이터 실측·새 학습 성적은 의도적으로 남은 공백이다.


# Claim–source ledger

| 주장 | 종류 | 근거 |
|---|---|---|
| 팀 최고는 e150+GI2 Public F1 0.833548 | 로컬 공식 원장 사실 | 저장된 제출·리더보드 계보 |
| I-ORS 80행 A/B는 파일 수준 조건부 효과만 식별 | 수학적 해석 | 두 제출 파일의 유일 차이와 aggregate F1만 관측 |
| type/boundary cascade는 pooled 음수 | 실행 사실 | `artifacts/p1_mstcn_type_boundary_cascade_shadow_20260829_v1/result.json` |
| frozen adapter는 두 창 모두 음수 | 실행 사실 | `artifacts/p1_mstcn_frozen_trust_adapter_shadow_20260829_v2/result.json` |
| partial pooling은 Q2 양수, Q3 음수, Q4 no-op | 봉인 실행 사실 | `artifacts/p1_mstcn_partial_pooling_calibrator_shadow_20260829_v1/result.json` |
| external I-ORS point residual은 큰 음수 | one-shot outer 사실 | `artifacts/p1_iors_external_point_residual_oof_v1/20260821T174744+0900/result.json` |
| MS-TCN++ 다단계 refinement는 과분할 완화를 목적으로 함 | 1차 논문/공식 구현 | https://github.com/sj-li/MS-TCN2 |
| ASRF는 boundary branch를 결합 | 1차 논문 | https://arxiv.org/abs/2007.06866 |
| 시계열 domain adaptation은 representation과 alignment를 함께 고려 | 1차 논문 | https://proceedings.mlr.press/v202/he23b.html |
| KORS QC에는 관측소·수심·시간 문맥이 중요 | 도메인 1차 연구 | https://e-opr.org/articles/xml/rQWX/ |
| environment-robust training이 다음 1순위 | 증거 기반 추론 | 후처리 3축의 transport 실패와 target/source covariate shift 감사 |

공식 점수 기대값, I-ORS 80행의 개별 TP 수, 대회 전체 최고 모델 구조는 현재 증거로 주장하지 않는다.

# 재생성 우선 실행 해석과 진행 경계

사용자 요청과 `docs/ocean_v2_codex/ROADMAP_after_codex_review_20260905.md` §5를 따른다. 원본 로드맵은 수정하지 않는다. 이 문서는 잘못된 규정 확대 및 과거 결과의 소급 재분류를 막기 위한 실행 해석이다.

## 우선순위

1. 기존 폴더를 삭제하지 않고 **새로운 빈 03_model**에서 배포 데이터로 학습한 뒤 새 프로세스로 05_answer를 생성한다. 실패 산출물·lock은 보존한다. 기존 답안이나 모델을 입력으로 사용하지 않는다.
2. 결과를 확인한 뒤 다음 실험의 평가 계약을 코드로 고정한다. 현재 v4를 새 로드맵의 반기/8블록/6분기 실험으로 재명명하지 않는다. 계약 고정 이전에는 다음 성능 실험을 시작하지 않는다.
3. 평균 변화, 불확실성, worst-block 및 slice 위험을 분리 기록한다. 결과를 본 뒤 새 0.8 확률 조건으로 과거 성공/실패를 바꾸지 않는다.
4. 기존 P2 T5 전문가 3-seed 실패 기록을 유지한다. 현재 프로파일 잔차 가설은 그 재시도가 아니다.
5. 최종 README와 상수 계보 감사는 실제 train→inference 계약을 반영해야 한다. 로컬 재생성, 저장 모델 replay, 독립 환경 패키지 재현 및 대회 적격성은 각각 다른 주장이다.

## 원본 문서에서 그대로 적용하지 않는 항목

- 숫자 리터럴을 모두 물리 상수로 제한하는 §1(c)는 운영진 규정보다 강하다. 정상적인 seed·구조·사전등록 hyperparameter는 금지되지 않는다. 반대로 리더보드 역산 계수를 JSON으로 옮겨도 적격해지지 않는다. 근거는 [최상위 정책](../../00_ORGANIZER_DATA_POLICY.md)의 9월 2일 공지 확인 기록이다.
- 과거 `official_final_submission_20260905` 패키지의 router_anchor/gi_spike2/bin17/Public alpha 체인은 현재 Sep5 clean control과 동일하지 않다. 과거 파일은 증거로 보존하되 새 학습·추론의 입력에서 제외한다.
- P1 MS-TCN, P2 v52+PAVA, P3 CPU 등가 평균으로 바꾸는 것은 별도 모델 변경이다. 기존 clean O/B, v23 blockmask, 배포 데이터 CatBoost+학습 router가 실제로 재생성된다면 이를 규정 때문에 자동 폐기할 이유는 없다. 사전등록한 P3 shrink 및 Ridge regularization도 Public 역산 alpha와 출처가 다르다.
- §2의 새 평가 분할·가중식·onset 정의는 현재 v4의 사전등록 계약이 아니다. 해당 날짜별 지원 표본, purge 후 train 충분성, 가중치 근거를 점검하고 새 버전으로 고정해야 한다. 과거 노출된 기간을 fresh라고 부르지 않는다.
- bootstrap 개선 비율은 선택 편향과 미관측 계절을 반영한 공식 점수 개선 확률이 아니다. 결과표에 기재하더라도 기술적·서술적 지표로 해석한다.
- 포털 제출물 삭제, 최종 모델 lock, 커밋/푸시는 코드의 자동 부수효과로 넣지 않는다. 별도 실제 권한과 결과를 확인해야 한다.

## 현재 원장

- P1 T–S: [실험 결과](../p1_ts_disagreement_20260905_v4/report-source.md), [재생성 v4 실패](../p1_clean_regeneration_20260905_v4/report-source.md).
- P2 프로파일 보정: [실험 결과](../p2_profile_copula_residual_20260905_v4/report-source.md), [재생성 v4 기술실패](../p2_clean_regeneration_20260905_v4/failure-report.md).
- P3 바람 증강: [실험 결과](../p3_wind_only_dropout_20260905_v4/report-source.md). clean 기준 재생성은 별도 v4에서 진행한다.

기술 정정은 새 ID로 기록한다. 성공할 때까지 임계값을 강제하거나 같은 lock으로 재시작하지 않는다. 이 문서 자체는 재생성 PASS나 최종 제출 준비 완료를 인증하지 않는다.

# P1 고정 tree ∪ MS 결합 오류 감사

결론: **추가 학습 없이 오류 집계를 완료했다. 두 모델이 함께 놓친 1,448행의 89.3%는 drift/offset이며, 단순 경계 보정보다 긴 구간 내부의 누락이 크다. 기존 b2f17 답안·모델·정책은 그대로 보존한다.** 이는 해당 완성 정책에 대응하는 과거 cross-fit 진단이며 공식 b2f17의 오답 분석이 아니다.

## 동일 분모의 결과

Q3/Q4 원 fold 소유와 287,862개 전체 키를 유지했다. 양성 10,685, 정상 277,177, 누락·중복·정답 불일치·사후 제외 0, 달력 경계 밖으로 이어지는 원 소유 행 119개도 유지했다. 고정 결합은 TP 9,237 / FP 447 / FN 1,448, F1 **0.9069664686533457**이다.

| 진단 | 실측 |
|---|---|
| 두 모델 모두 놓친 FN 유형 | offset 696, drift 597, mixed 95, noise 51, spike 8, flatline 1 |
| FN 연속 구간 | 총 33개. 이 중 24–72시간 길이 4개가 1,108행(76.5%) |
| 양성 구간 경계/내부 | 처음·마지막 2시간 경계 FN 201 / 내부 FN 1,247(86.1%) |
| 평가 partition 경계/내부 | 경계 FN 5·FP 0 / 내부 FN 1,443·FP 447 |
| 정상 행 FP 출처 | tree만 156 / MS만 279 / 양쪽 12 |
| FP와 실제 양성의 거리 | 2시간 이내 FP 128 / 그 밖 FP 319 |
| FN 집중 정점·층 | I-ORS/1 401, G-ORS/1 276, I-ORS/2 259, S-ORS/2 234, I-ORS/5 165 |

전체 양성 연속구간 100개 중 완전 누락 11개, 부분 누락 17개, 전부 탐지 72개다. 정점·층 전체, 유형×길이, 유형×경계, FP 연속구간 등 완전한 집계는 [result.json](../../artifacts/p1_composed_error_audit_20260906_v1/result.json)에 있다. 특정 정점·층을 고정 예외처리하거나 정상 급변을 제거하라는 근거는 아니다.

## 해석과 한 가지 후속 가설 — 실행하지 않음

다음 독립 비교 후보는 **기존 MS-TCN의 양성 유형별 학습 손실 균형만 조정하는 한 가지 정책**이다. 기존 type head를 새로 발명했다고 하지 않고, 현재의 row/event/type 학습에서 offset·drift의 지속 구간을 더 잘 학습할 수 있는지 확인한다. 정상 행은 삭제·하향가중하지 않고 기존 정상 손실과 표본을 유지한다. 구체 가중 규칙은 각 fold의 학습 유형 빈도로만 결정·봉인하고 현재 Q3/Q4의 오답 위치/정점/날짜를 학습 입력이나 예외 규칙으로 쓰지 않는다.

현재도 [type BCE 보정](../../src/p1_qc/ms_tcn_asrf.py:566)은 있다. 다만 event-positive 행의 다섯 head 전체를 한데 모아 capped negative/positive 비율 하나를 적용한다. 따라서 새 제안은 “불균형 보정이 전혀 없었다”가 아니라 **유형별 정규화와 기존 pooled 보정의 독립 비교**에 한정한다. 빈도 균형이 실제 난이도 부족을 해결한다는 보장은 없고 희소 spike 과대가중 위험도 있으므로, 가중 규칙·상한·평가 단위가 명시된 새 계약 전에는 실행할 수 없다.

검증 시 tree·특징·decoder·MS threshold를 고정하고 **실제 tree ∪ 새 MS의 전체 동일 키 pooled F1**을 기준 결합과 비교해야 한다. type-only recall이나 MS 단독 개선으로 승격하지 않는다. 이 분석은 가설 생성용으로 이미 노출된 retrospective 자료이고, 신규 일반화 증거가 아니다. 추가 fit/가중치 구현/후보 생성은 0이며 portable 패키지를 먼저 완성한다.

이미 실패한 [T–S 19특징](../p1_ts_disagreement_20260905_v4/report-source.md), [core rounds/learning-rate→MS bridge](../p1_core_tuning_ms_bridge_20260906_v1/report-source.md), [범위·셀 후처리](../p1_trainfit_postpolicy_20260906_v1/report-source.md)를 재실행하는 제안이 아니다. 현 MS 원형은 이미 subtype head와 2,048행 window를 갖추므로 “type head 추가” 또는 “긴 context가 전혀 없었다”를 새 발견으로 주장하지 않는다.

## 정의·검증·접근 범위

- 유형은 배포 train의 `anomaly_type`만 사용했다. 중복 동일 atom은 하나로, 서로 다른 atom은 mixed로 묶었다. 행별 유형과 한 binary-positive run의 혼합 유형은 다른 분모다.
- 양성 이벤트는 전체 배포 이력의 정점·층별 정확한 10분 연속 label=1 run이다. 생성기 event ID가 아니며 인접 주입은 합쳐질 수 있다. 정상과 양성의 2시간 거리는 **진단용 정답 정보**이며 inference 특징이 아니다.
- 유형별 양성-only 집계에서 FP=0은 정상 분모가 없기 때문이지 공식 FP=0 보장이 아니다. FN 비율과 정상 FP 비율을 분리한다.
- [스크립트](../../scripts/p1_composed_error_audit_20260906_v1/run.py), [합성 10 PASS](pytest.xml), Ruff PASS. [독립 QA](../../artifacts/p1_composed_error_audit_20260906_v1/independent-qa.json) **83/83 PASS**: sklearn confusion/F1, scalar 전체 run 길이, 원본 키·fold·label, component attribution 및 event/burst 분모 검산.
- 실행 10.359초, CPU1/GPU0, fit0. train 키·label·anomaly_type와 허용된 historical OOF만 읽었다. 관측값 temp/psal/depth·모델·공식 입력·hidden·CSV·업로드·Git 쓰기·attempt lock 생성 모두 0, 원시 행 출력 0.
- SHA256: script `98023fad4a07db32ac13cbf8fc87d1f0f54d354f6af02cc2ae8cc844b64f1076`; result `d1be45013548c55d73a22ec4c01a64d2a8f3a62306875ec8906e3b987bedd558`; QA `9e3ff924b58919659c6a2a46f5fbb5e2e07e4f040356eb7ce6d3811e09f7eb4e`.

# 추가 검증·개선과 최신 후보 패키징

사용자 승인: 2026-09-06 “진행하세요”. 새 학습·내부 분석·패키징을 병행한다. 기존 완료된 76fit 사이클, 후보 CSV, 봉인 코드와 소비된 lock은 변경하거나 다시 실행하지 않는다. Git commit/push/최종 모델 잠금은 이번 작업에 포함하지 않는다.

## 사전 작업 계약

| 담당 | 신규 작업 | 예산·판정 |
|---|---|---|
| P1 | 고정 tree union OR MS의 같은 키 FN/FP 구간 분석; b2f17 portable cold/saved 패키지 | 분석 fit0/CPU1. cold 재학습은 새 경로·새 타이머·GPU 별도 배정 후 ≤6h. 기존 공식 점수 승계 금지 |
| P2 | C60/L120 나머지7fold×2seed×2arm의 다중 seed 증거 보완 | 신규28fits/CPU2/GPU0/60분. B3 Sep–Oct 주평가 유지, 전체8fold3seed는 보조 평가. 기존 B3/first-seed exact-hash 재사용 |
| P3 | 현재 numeric-lead 코어의 학습률/학습량 소규모 비교 | 최대3정책(대조 포함), 최대24 신규 historical fit/60분, earlier-inner 선택과 outer 평가 분리. 정확한 계약은 실행 전 담당 seal로 고정 |
| Root | 기존/신규 후보 패키지 경계·계보·독립 QA·사용 안내 | cold와 saved replay/공식 채점을 분리, 원시 행 출력0 |

GPU 큐: P2 추가28fit → P3 소규모 비교 → P1 portable cold(MS 포함). 준비/합성 검사/CPU 분석은 병행하되 CPU 총량을 제한한다. P2/P3 새 후보 cold 검사는 담당 완료 후 별도 배정한다. 새 학습 중 중간 성능으로 범위·seed·gate를 변경하지 않는다.

16:06 KST 자원 배정 보충: P2 completion의 전체 replay/QA 완료 뒤 약3분인 L120 cold 3fit을 먼저 실행하고 P3 tuning으로 넘긴다. P1 cold는 Q4 earlier-inner 4 + full tree 4 + MS full 3 =11fit의 새 배포 재생성 계약이며 기존31fit 봉인 재실행이 아니다. P3 source-only prepare는 별도 외부 ZIP 추출에서 CPU2/fit0으로 시작했고 전체6h 타이머도 이 시점부터 시작한다. P3 cold의 17fit은 GPU를 별도로 배정하기 전에는 실행하지 않는다.

16:20 KST 자원 보완: P3 tuning v1은 성능 열람 전 첫 baseline inner2fit의137.049초 실측을 기반으로 전체3941.581초를 예측해 원3600초 예산에서 RESOURCE_STOP했다. 이는 과학적 실패가 아니다. v1 source/config/lock/모델/receipt는 보존한다. 새 `p3_numeric_training_curve_budget90_20260906_v2`는 시간만5400초로 보완하고 이2개 모델을 hash/native parameter 검증 뒤 재사용한다. 같은 recipe/선택/seed/분할/특징과 총20fit 범위를 유지해 최대18신규fit만 가능하다. baseline 선택 시 추가inner4fit으로 끝난다. Multi의 사후 수용 hash와 원래 save-time receipt 부재를 구분한다. 새 계약 봉인 및 root GPU 배정까지 예측/학습0. GPU 큐는 P1 saved+11fit cold → P3 원 numeric cold17fit → P3 budget90 보완안 순서다. 성능값에 따른 탐색 확대나 원 시도 재시작이 아니다.

## 유지할 판단

- 평균 개선 우선, 특정 Q/층/리드 악화와 CI는 별도 위험표이지 자동 veto가 아니다.
- 전체 pooled RMSE는 SSE/총행수에서 계산한다. P2 기존 가을 주평가를 잘 나온 다른 계절로 바꾸지 않는다.
- 이미 반복 노출된 역사 구간은 fresh holdout이 아니다. 확대 seed 검증도 새 독립 기간을 만든 것이 아니다.
- 공식 입력은 내부 QA를 마친 고정 후보의 허용된 추론/키 검증 단계에서만 사용한다. hidden/외부 관측/리더보드 역산0.
- 최신 P1 b2f17, P2 fee6118, P3 ff42a6 후보는 현재 미채점이다. 기존 fallback ZIP과 답안은 별도 보존한다.
- Chrome `Debugger unattached`와 차단된 재시작은 우회하지 않는다. 브라우저 차단 때문에 독립 로컬 작업을 멈추지는 않는다.

## 완료 산출물

문제별 canonical report/수치·핏 수·시간·QA, source-only cold ZIP과 saved ZIP의 역할/실행 명령/manifest, 실제 새 경로 cold·답안 replay 결과, 미완료 공백표를 연결한다. 준비만 된 패키지를 전체 재학습 완료로 표시하지 않는다.

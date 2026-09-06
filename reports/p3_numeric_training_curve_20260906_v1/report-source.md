# P3 numeric training curve v1 — 자원 한도 중단

결론: **RESOURCE_STOP이며 과학적 NO_GO가 아니다.** 첫 예정 baseline 2 fits는 완료했지만, 결과 열람 전 고정 비용식의 예측이 3,941.581초(65.693분)로 60분 한도를 초과했다. 학습률 비교의 승패, RMSE 개선, 예상 공식 점수는 아직 판단할 근거가 없다. 기존 numeric 후보 `ff42a6a0…37960`은 불변이다.

- [원 사전등록](preregistration.md), [사전검증](preflight.json), [자원 측정](resource-pilot.json), [원 중단 receipt](execute-failure.json), [독립 자원 QA](resource-stop-qa.json).
- 실제 성공 backbone 2 fits, router 0. Single 113.877625초, multi 23.171393초, 합계 137.049017초다. Multi 시간은 저장된 합계에서 single receipt를 뺀 값이다. 원 attempt lock부터 중단 receipt까지 138.603629초다.
- 봉인 비용식은 `1.2 × 첫 두 fit 시간 × (3.25 + 1.5 × 64,416 / 4,664)`다. 이는 실제 완료 시간이나 품질 추정이 아니라, 최대 분기의 row-linear 보수적 자원 예측이다. 측정값을 본 뒤 식/한도를 완화하지 않았다.
- 실제 child PID 40488 / launcher 28068은 종료됐고 GPU를 해제했다. 선택 JSON, outer-start, 신규 OOF 및 과학적 result는 생성되지 않았다. 공식/hidden/CSV/upload 접근은 0이다.

## 무결성 및 증거 한계

CPU에서 두 저장 CBM을 native load만 수행했다(예측·재학습 0). Single은 700 trees / 594 native 입력 열, multi는 1,200 trees / 592 native 입력 열이고 둘 다 categorical index `[0]`이다. 591 공통 수치 특징 외 single에는 station/lead/current residual 열, multi에는 station 열이 추가되는 기존 구현이다.

Single SHA `e26e30cb5c0daf6c3e49f09276097dcc9d0342c7b0ec305fd5c796eeb9b8b811`는 원 fit receipt와 일치한다. Multi SHA `f7197f8aea417642fb51047d2f69dc0c4f109d858f1c812d154d990c04913ff2`는 **중단 후 acceptance-time hash**다. 개별 multi receipt/prediction hash는 예외 전 메모리에만 있었으므로 save-time 무결성 증거로 과장하지 않는다.

Progress와 fit-receipts는 각 fit 직전에 저장되어 마지막 값이 1이다. 이를 실제 1-fit으로 잘못 세지 않으며, 완료된 2-fit pilot 및 native 1,200-tree multi 저장물을 함께 근거로 2 successful fits를 기록했다. 원 progress/receipt/runner/config/lock은 수정하지 않았다.

독립 자원 QA는 18/18 PASS, 기존 합성 16 tests와 Ruff PASS다. 이는 비용식·봉인·저장 상태 검사이며 성능 검증이나 saved-prediction exact replay PASS가 아니다.

## 다음 분기

Root가 성능 미열람 상태에서 측정 시간만을 근거로 별도 `p3_numeric_training_curve_budget90_20260906_v2` 자원 보완을 승인했다. 원 v1을 재시작하지 않는다. 새 시도는 같은 3 recipes / held-inner / 5 outer / seed / 특징 / 정책을 유지하고, 원 2 models를 exact hash 및 native metadata로 검증한 뒤 최대 18 fits만 추가하도록 별도 봉인한다. Multi의 사후 hash 한계도 승계한다. 실제 예측·학습은 새 봉인과 별도 GPU 배정 후에만 가능하다.

기존 numeric whole-cold 12 backbone + 5 router 패키지 검증은 별도 작업이며, 이 자원 중단 실험을 whole-cold 완료 또는 제출 후보로 표시하지 않는다.

# P1 수심 메타데이터 스트레스 — 별도 0-fit 진단

## 상태: 계약·합성검사 완료, 실측 진단 시작 전

Root 승인 범위에서 이미 완료된 양측 OOF와 동일한 outer 모델·정책·threshold·rules·키·label·달력·관측 수심을 유지한다.
`nominal_depth_m`, `depth_regime` 두 열만 바꾼다. 원 양측 결과를 수정하거나 새 정책을 고르지 않는다.

1. 원래 year-key 통계의 표준 양측 OOF를 exact 재현한다.
2. 두 메타 열을 미지원 연도 상태인 NaN / station별 unknown layer 범주로 바꾼다.
3. 각 모델 train_stats의 station-layer별 유한 연도 median들을 같은 가중치로 median한 뒤 기존 2m rounding을 적용한다.
   이 한 가지 fallback만 고정한다. 미지원 셀은 NaN/unknown을 유지한다.

CPU2 / GPU0 / 새 모델fit0 / 공식·hidden·CSV·upload0 / wall cap10분이다. 성능값으로 fallback을 재선택하지 않는다.
별도 프로세스의 저장 확률→고정 decoder→키·label·OOF 산식 QA는 모델 전체 재학습·cleanroom 재현과 구분한다.

## 먼저 확인한 실제 소스 동작

`run_p1_score_repair_20260905_v1.stats_fit`은 `(station, year, layer)`를 깊이 통계 키로 사용한다.
`run_p1_depth_contract_repair_20260905_v2.features(current_depth=False)`에서 lookup miss가 나면 숫자 NaN과
`station|unknown|l...`가 된다. `TabularEncoder`는 numeric NaN을 그대로 유지하고 unseen category를 -1로 바꾼다.
median/0 대체가 아니다. `depth_raw`와 `depth_missing`은 실제 관측 수심/결측을 그대로 유지한다.

원래 소스 함수에 동일한 합성 관측을 넣고2025→2026년을 바꾼 결과 **14/14 differential checks PASS**:
두 메타 열만 변하고 다른78열은 exact, 실제 raw depth는 남는다. synthetic pytest6 PASS, Ruff PASS.
알려진 연도 하나의 fallback은 기존 메타를 복원하며, 여러 연도의 equal-year median 및 미지원셀도 합성검사했다.
정상 train min/max 바깥의 미래 정상값 또는 그 안의 이상값이 가능한 toy로 FP0/FN0 보장 불가도 명시한다.

## 해석 범위

- 표준 양측 OOF, forced unknown-year, frozen-training-stat fallback을 세 표로 분리한다.
- 이미 본 과거 label을 쓰는 반사실 진단이다. 미래/공식 성능 증명이 아니다.
- 모델은 원래 year별 메타로 학습된 그대로다. fallback을 사용해 다시 학습한 정책의 평가가 아니다.
- 과거 depth-info 공식 실패는 old Q4-inner union vs final-inner balanced/.2/OFF라는 완성정책 차이도 포함했다.
  수심 단독 효과나 이 fallback이 나쁘다는 인과결론으로 사용할 수 없다.
- 지금도 실제 관측 수심 누락은 유지한다. 두 메타 열 외 배포 분포 차이는 이 진단에 포함하지 않는다.

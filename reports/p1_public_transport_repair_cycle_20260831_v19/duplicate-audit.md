# P1 v19 duplicate and leakage audit

## 결론

`P1_1_GORS_CAUSAL_ONE_STEP_RUN_EXTENSION`은 실행 가능한 단일 development 후보다. 다만 연산자 자체는 v12의 전역 one-step trailing dilation과 같고 support만 `G-ORS`로 고정하므로 `HIGH_SEMANTIC_REUSE`다. exact v12를 새로운 계열로 포장하지 않으며, 이번 Q3/Q4도 fresh confirmation으로 주장하지 않는다.

## 고정 차이

- v12: 모든 station-layer의 frozen reference-positive run 뒤 한 행을 추가했다.
- v19: `G-ORS`에서만 같은 연산을 수행하고 `I-ORS`와 `S-ORS`는 bit-exact다.
- 원본 reference bit에만 lag를 계산하며 새로 추가된 행을 다시 확장하지 않는다.
- station, 10분 cadence, 1행 span, gate는 결과 뒤 변경하지 않는다.

## 근거와 한계

- v12에서 G-ORS 단일층 ΔF1은 양수였지만 다른 station-layer가 전체 손실을 만들었다.
- frozen official factorial에서 G-ORS e150 15행 제거는 표시 Public F1을 `0.004519` 낮췄다.
- 두 사실은 G support를 선택한 방향 근거일 뿐 Public row label이나 v19의 독립 검증이 아니다.
- negative-evidence registry의 global topology/union 실패를 보존한다. v19 실패 시 G-only exact 조합도 닫고 cadence/span/station을 결과 기반으로 바꾸지 않는다.

## 비중복 경계

- v17 MiniRocket은 512개 fixed convolution PPV 표현과 logistic head를 2회 적합한다.
- v18 soft symbolic transition은 PAA/soft-symbol transition 표현과 logistic head를 2회 적합한다.
- v19는 표현 학습과 모델 적합이 없고 frozen binary reference의 한 행만 결정론적으로 확장한다.

## 누출·접근 경계

- historical scoring은 frozen E150 OOF와 Q3/Q4 labels만 사용한다.
- Public aggregate는 support prior로만 기록하며 rowwise truth로 사용하지 않는다.
- strict internal PASS 전 official test/champion을 열지 않는다.
- hidden truth와 upload는 항상 0이다.

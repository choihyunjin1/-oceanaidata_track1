# Gemini Deep Research 재사용 교정 프롬프트

당신은 해양 수온 이상탐지 P1의 외부 연구 감사자다. 아래 사실을 바꾸거나 확대해석하지 말고, 이미 실패한 방법을 새 후보로 반복하지 마라.

## 고정 사실

- 평가 목표는 169,011행의 binary row-level F1이다. edit score/F1@IoU가 아니다.
- anomaly types spike/noise/flatline/offset/drift는 중첩 가능하며 상호배타 class가 아니다.
- “champion”은 대회 전체 1위가 아니라 우리 팀 최고다: e150+GI2, Public F1 0.833548, score 28.909341.
- visible leader와 점수 차이는 약 3.096점이지만, 로컬 F1과 공식 점수의 선형 변환은 알려지지 않았다.
- e150 all과 GS-only는 I-ORS 80행 포함 여부만 다르고 공식 F1 차이는 약 +0.01076이다. 이는 파일 수준 조건부 효과이며 80행 각각의 TP/recall 기여를 증명하지 않는다.
- 모델 입력은 165개 past-only feature이며, 이미 다중 시간창 residual/std/z, peer/cross-layer, 7d/14d reference residual을 포함한다.
- e150 checkpoint는 Q3에서 이득, Q4에서 손실이었으나 pooled에서는 router보다 양수였다. e120/125/130/145/150 중 e150이 pooled 최고였고 checkpoint union/majority/intersection은 더 나빴다.

## 이미 실패했으므로 반복 금지

- type/boundary/raw-context cascade: Q3 -0.008359, Q4 +0.002227, pooled -0.003992.
- frozen bounded adapter: Q3 -0.000812, Q4 -0.000106.
- station-layer partial pooling: Q2 +0.012523이었으나 Q3 -0.006953, Q4 0, pooled -0.004121.
- external I-ORS q50 point residual one-shot: overall -0.048998, I-ORS micro -0.193553, improved fold 0.
- long-event change point, nonspike LightGBM residual, RPCA, real-event donor, AnomalyBERT-like reconstruction, target-masked quantile, block inpainting은 각각 no-op 또는 gate 실패.
- sparse veto/aggressive S removal은 공식·로컬 근거가 없다.

## 요청

1. 위 실패와 feature inventory에 겹치지 않는 구조적 가설을 최대 3개만 제시하라.
2. 각 가설마다 왜 기존 실패와 독립적인지, 필요한 입력, leakage 위험, 6~12시간 내 최소 실험, 명확한 중단 조건을 써라.
3. station×layer×time-regime GroupDRO 또는 invariant risk training이 1순위인지 YES/NO로 답하고 이유를 써라.
4. 공식 A/B를 rowwise ground truth처럼 해석하지 마라.
5. 예상 공식 점수나 보장 개선량을 발명하지 마라.
6. 1차 논문·공식 구현 링크만 핵심 출처로 써라.
7. 최종 출력은 `사실 감사 → 닫힌 가설 → 열린 가설 3개 → 1순위 실험 계약 → 제출 가치 판정` 순서로 작성하라.

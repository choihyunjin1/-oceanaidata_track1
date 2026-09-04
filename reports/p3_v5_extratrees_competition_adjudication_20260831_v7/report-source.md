# P3 v5 ExtraTrees competition adjudication v7

## 결론

- **Scientific: INCONCLUSIVE.** CI90이 0을 교차하므로 강한 과학적 challenger로 승격하지 않는다.
- **Competition: EXPECTED_VALUE PASS / READY_NOT_UPLOADED.** 만료성 제출 슬롯의 정보가치를 포함한 별도 행동 판정이다.
- frozen v5 ExtraTrees를 seed 20261832로 정확히 1회 full-fit했고 1,200행 비중복 CSV를 만들었다.

## 내부 근거와 점수 범위

- delta RMSE(candidate-reference): -0.004558558m
- episode bootstrap CI90: [-0.009923488, +0.000795217]m; P(improve)=0.9156
- 예상 점수 변화 conservative/central/optimistic: -0.012621 / +0.072348 / +0.157493
- heuristic probability-weighted action points: +0.065176
- 보수 시나리오는 손실이며 실제 official 수송은 과거 방향 반전 때문에 부호도 보장되지 않는다.

## 구조 QA

- CSV SHA-256: `1bb1a90c149e566497f95fcb9d1bb1aa3895f4fef341afc6b30d6fe6710ca65d`
- champion 대비 changed rows: 400; short leads exact no-op: True
- finite/domain: min=0.733751915m, max=4.163248644m
- hidden truth 0, upload 0.

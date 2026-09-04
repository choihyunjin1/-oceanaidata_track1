# Gap matrix

| Gap | Current control | Remaining limitation |
|---|---|---|
| 다른 family 잔차의 비교환성 | exact family → tier → global precedence | family별 공식 표본이 대부분 1개 |
| 결과 후 family 재분류 | 사전등록 필드와 소급 금지 | 수동 분류 판단은 독립 QA 필요 |
| unseen conditional router 낙관 | global 0.321905690 penalty | 높은 기준 때문에 실제 돌파 후보가 적음 |
| Public score leakage | family penalty 외 학습 사용 금지 | 제출 수가 적어 transport CI 추정 불가 |
| penalty 완화 남용 | same-family adverse max-only, n>=3 전 완화 금지 | 독립 pair가 쌓일 때까지 empirical guardrail |

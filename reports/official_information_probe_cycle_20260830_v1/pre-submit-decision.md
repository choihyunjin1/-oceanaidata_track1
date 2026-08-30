# 2026-08-30 공식 정보 probe 제출 전 결정서

## 결론

오늘 남은 슬롯 `P1 3/3`, `P2 2/3`, `P3 2/3`에 맞춰 총 7개 후보를 **새 점수 관측 전에 모두 동결**했다. 각 후보는 현재 공식 최고 제출을 크게 바꾸는 새 모델이 아니라, 이미 공식적으로 일부 효과가 확인된 축을 station·season-bin별로 분해하는 정보 probe다. 정확한 과거 제출 벡터와의 중복, schema·행·키·순서·finite/domain을 검사했으며 모두 `READY_NOT_UPLOADED`다.

새 제출이 악화하더라도 기존 문제별 최고 제출을 대체하기 위한 후보가 아니라 공식 기여도를 식별하기 위한 probe다. 단, 일일 슬롯은 실제로 소모되며 일부 후보의 하방은 작지 않다.

## 공식 규정과 시간 경계

- 공식 참가자 전용 공지는 답안 업로드를 `팀당·문제당·하루 3회`로 제한한다.
- 세 문제 점수는 합산된다.
- 실제 코드·가중치의 최종 모델 제출 기한은 `2026-09-07`이며, 모델을 최종 제출하면 이후 답안 CSV를 추가 업로드할 수 없다.
- 따라서 오늘은 답안 CSV만 사용하고 `모델 최종 제출하기`는 누르지 않는다.
- 2026-08-30 22:38 KST 공식 문제 페이지 실측 잔여량: P1 `3`, P2 `2`, P3 `2`.

공식 근거: [대회 제출 안내](https://oceanaidata.org/app/notices), [P1](https://oceanaidata.org/app/problems/5), [P2](https://oceanaidata.org/app/problems/6), [P3](https://oceanaidata.org/app/problems/7).

## 고정 제출 순서

### P1 — e150 station ablation, 3건

현재 공식 최고 F1 `0.833548`을 기준으로 GI2·anchor는 보존하고 e150 추가행만 station별로 제거한다. F1은 비선형이므로 세 점수는 단순 합산하지 않고, 네 개 고정 예측집합의 차이 및 difference-in-differences로 해석한다.

1. `P1 e150 G-station ablation`
   - SHA-256 `59d862b62b24a03c637f58a676d9dc99186f148a6f710f9d0cd3f162f6ca2ce8`
   - 169,011행, champion 대비 G-ORS 15개 양성행 제거
   - 질문: G-ORS e150 추가가 공식 F1에 순기여하는가?
   - 하방: 세 후보 중 가장 작다.
2. `P1 e150 S-station ablation`
   - SHA-256 `9f5e8ca7a19d17b317e969fb152f2cc77f50fb5d1cc53757de200b68be720983`
   - 169,011행, champion 대비 S-ORS 238개 양성행 제거
   - 질문: e150 운송 효과의 대부분이 S-ORS에서 발생하는가?
   - 하방: 크다. 과거 broad sparse removal은 공식 F1이 `-0.013209` 악화했다.
3. `P1 e150 I-only transport`
   - SHA-256 `e6a54b0aaf6bf6227362a2706447e4c09bfe80c8510248c2206f204f1ad5c042`
   - 169,011행, champion 대비 G/S 253개 양성행 제거
   - 질문: I-ORS e150 효과 및 G×S interaction이 공식 표면에서 재현되는가?
   - 하방: 세 후보 중 가장 크다.

Manifest/QA: `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260830_P1_STATION_ABLATION_INFORMATION_PROBES_READY_V1/SET_MANIFEST.json`, `INDEPENDENT_QA.json`.

### P2 — rank-1 season-bin decomposition, 2건

현재 공식 최고 RMSE `0.430209`는 alpha50 `0.431252` 대비 bin17·18 rank-1 보정을 함께 사용한다. 두 후보는 하나의 bin만 남겨 각각의 공식 순효과를 분리한다. 직전 Gaussian 전체 probe `0.442259`의 방향으로 양의 blend하면 작은 `alpha=0.1`부터 공식 RMSE 악화가 예측되어 폐기했다.

1. `P2 rank-1 계절 bin17 단독 분해`
   - SHA-256 `99c6925cec605905c80f2924c5655b3dd83ed712c9f27853c58d6d9e0f74e2e2`
   - 26,061행, champion 대비 762행 변경, RMS `0.018519°C`
   - 질문: bin18 보정이 공식 전이를 해쳤는가?
2. `P2 rank-1 계절 bin18 단독 분해`
   - SHA-256 `0d213e97b9435862bbc892ac358afdc99b0a8834740915720f96bd420761d557`
   - 26,061행, champion 대비 2,464행 변경, RMS `0.023161°C`
   - 질문: bin17 보정이 공식 전이를 해쳤는가?

두 local cross-fit bin의 CI90은 모두 개선 방향이었지만 proxy/official transport 위험이 있으므로 성능 보장이 아니다. Manifest: `reports/p2_rank1_bin_decomposition_probes_20260830_v1/prepared-probes.json`.

### P3 — KMA station ablation, 2건

현재 공식 최고 RMSE `0.575233`의 18·24h uniform KMA `alpha=0.425` 보정에서 station 하나만 alpha=0 축으로 되돌린다. 이전 lead-split과 lead-continuous 실패 때문에 새 alpha를 외삽하지 않는다.

1. `P3 KMA 42.5% station ablation · S-ORS 제외`
   - SHA-256 `a5a16ba207ed1cccf16383e1de7b932417666917b0eb2b9c54a00fdb7ab67351`
   - 1,200행, S-ORS 18·24h 120행 변경, RMS `0.068528m`
   - 질문: S-ORS 보정의 공식 기여는 양수인가?
2. `P3 KMA 42.5% station ablation · I-ORS 제외`
   - SHA-256 `868d18d7a2d62d49b6d97712e686db7a55bbb16e38cea2659584a0c397275f4f`
   - 1,200행, I-ORS 18·24h 140행 변경, RMS `0.053429m`
   - 질문: I-ORS 보정의 공식 기여는 양수인가?

두 새 점수와 이미 알려진 alpha=0/전체 alpha=.425 점수를 제곱 RMSE 공간에서 결합하면 G/I/S의 Public MSE 기여를 분해할 수 있다. Manifest/QA: `submissions/p3_20260830_station_ablation_probes_v1/FREEZE_MANIFEST.json`, `QA_RECEIPT.json`.

## 제출 후 해석 계약

- 7개 모두 score 관측 전에 동결됐고 중간 결과를 보고 파일·순서·값을 바꾸지 않는다.
- 점수가 좋으면 해당 고정 후보를 문제별 Public challenger로 기록한다.
- 점수가 나쁘면 동일 축의 nearby 값 재탐색 근거로 쓰지 않고, station/bin별 공식 음성 근거로 원장에 남긴다.
- 모든 결과는 Public 근거일 뿐 Private 성능 보장은 아니다.
- 최종 모델 제출은 하지 않는다.

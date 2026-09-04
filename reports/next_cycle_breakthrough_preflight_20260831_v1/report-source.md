# 다음 돌파구 사이클 실행 전 판정

## 기술 요약

**결론은 `NO_NEW_ONE_SHOT_AUTHORIZED_UNTIL_NEW_INFORMATION`이다.** 이번 사이클에서는 새 모델을 학습하지 않았다. P1은 미사용 local label tail이 0행이고, P2는 같은 세 historical block과 같은 Public feedback에 반복 적응했으며, P3는 KMA 0.425·station 제거·ERA5 Hs² 잔차까지 공식적으로 방향이 판별됐다. 지금 새 fit을 추가하면 새 성능 증거가 아니라 이미 노출된 선택 기준에 대한 추가 최적화가 된다.

- **P1:** `BLOCKED_NO_FRESH_LABEL_SURFACE`. 최적 체크포인트는 보존하되, 노출된 Q2–Q4에서 다시 고르는 행위는 확인이 아니다.
- **P2:** `HOLD_BIN17_CHAMPION_NO_SAME_PUBLIC_ADAPTATION`. bin17-only가 alpha50보다 `0.001058°C` 개선됐고 bin18-only는 `0.000015°C` 악화됐다. 같은 Public 결과로 다른 bin을 고르는 것은 금지한다.
- **P3:** `HOLD_KMA_0425_CLOSE_KMA_AND_ERA5_EXACT_AXES`. uniform KMA 0.425는 alpha=0보다 `0.008659m` 개선됐지만 S/I station 제거는 각각 `0.003869m`/`0.003718m` 악화됐고, 과거 ERA5 Hs² residual도 당시 champion보다 `0.001846m` 악화됐다.

따라서 앞서 제안했던 P3 station별 강도 확장은 최신 공식 분해와 충돌하므로 철회한다. 다음 P3 가설은 KMA 미세조정이 아니라 여러 fresh storm episode에서 사전 고정한 forcing-error residual이어야 한다.

## 문제별 근거와 판정

### P1 — 최적 체크포인트는 자산이지만 새 성능 증거는 아니다

P1 validation audit의 보수적 하한은 outer-result 13회, candidate-fold 평가 5,265회이고, fixed Q4 이후 virgin local tail은 0행이다. 모든 776,706 train row도 deployment fit에 쓰였다는 별도 promotion audit와 일치한다. 따라서 과거 최고 checkpoint를 제출 자산으로 보존하는 판단과, 그 checkpoint의 성능을 새롭게 확인했다는 주장은 분리해야 한다.

이 판정은 체크포인트 선택 자체를 부정하지 않는다. 학습 중 가장 좋은 epoch를 저장하는 것은 맞지만, 그 epoch를 고른 validation surface로 다시 일반화 성능을 주장하면 선택 편향이 생긴다. Cawley와 Talbot은 noisy selection criterion의 반복 최적화가 model selection 자체를 과적합시킬 수 있음을 보였다. [JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

재개 조건은 단 하나다. 모델·feature·checkpoint·threshold·slice 선택에 한 번도 쓰지 않은 chronological label block을 먼저 봉인한다. P1 anomaly는 range/event이므로 point-only F1이나 event 전체를 채우는 shortcut 대신 range-aware precision/recall을 함께 고정한다. [Tatbul et al., NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)

### P2 — bin17은 유지하고 같은 Public surface에서 확장하지 않는다

P2의 2026-08-30 공식 분해는 disjoint support에서 명확하다. bin17 correction은 alpha50 대비 `-0.001058°C`, bin18 correction은 `+0.000015°C`였다. bin17-only가 현재 Public champion이다. 그러나 이 결과는 같은 Public split의 feedback이므로, 인접 bin이나 새로운 rank를 여기서 다시 고르면 독립 정보가 아니다.

Low-rank ocean reconstruction 자체는 계속 유효한 방향이다. DINEOF는 유효 EOF 수와 reconstruction error를 cross-validation으로 결정한다. 즉, 다음 rank/season 선택은 Public 결과가 아니라 사전 봉인한 same-season local block에서 해야 한다. [Beckers & Rixen 2003](https://doi.org/10.1175/1520-0426(2003)020%3C1839:ECADFF%3E2.0.CO;2)

재개 조건은 두 가지 중 하나다. (1) Public feedback과 무관하게 봉인한 같은 계절 block, 또는 (2) rank·season·transport rule을 그 block을 열기 전에 모두 고정한 새로운 mechanistic factor다. 이 조건 전에는 bin17 champion을 유지한다.

### P3 — KMA와 기존 ERA5 잔차의 정확한 축은 닫혔다

P3 official 결과는 uniform KMA 0.425가 alpha=0 대비 `-0.008659m` 개선됐음을 보였다. S-ORS와 I-ORS correction을 제거하면 각각 `+0.003869m`, `+0.003718m` 악화된다. 표시 RMSE rounding의 모든 16개 corner에서도 세 station 기여는 양수이고 순서는 S>I>G다. 과거 champion-matched ERA5 Hs² residual은 공식적으로 `+0.001846m` 악화됐다.

따라서 더 촘촘한 alpha, station 제거, 같은 ERA5 energy blend는 새 돌파구가 아니다. 다만 문헌은 wave-state와 wind covariate로 numerical wave-model residual을 학습하는 구조 자체가 가능함을 보여준다. [Ellenson et al. 2020](https://doi.org/10.1016/j.coastaleng.2019.103595) 다음 후보는 기존 KMA/ERA5 blend를 미세조정하는 것이 아니라, multiple fresh episode에서 검증 가능한 새 forcing-error target이어야 한다.

## 범위·데이터·지표 정의

- **P1 exposure 하한:** 과거 ledger에 기록된 outer-result/closed run과 candidate-fold 평가 수다. 독립 실험 수의 정확한 추정치가 아니라, 이미 충분히 노출됐음을 보이는 보수적 하한이다.
- **P2 공식 delta:** 같은 Public split에서 pre-frozen candidate와 comparator의 표시 RMSE 차이다. Private transport를 증명하지 않는다.
- **P3 공식 delta:** 각 candidate RMSE에서 그 candidate의 명시된 contemporary baseline RMSE를 뺀 값이다. 음수는 개선, 양수는 악화다.
- **fresh surface:** model, feature, checkpoint, threshold, slice, postprocess 선택 중 어느 단계에도 쓰이지 않은 label surface다.
- 이번 run은 aggregate JSON receipt만 읽었다. official test/sample/submission row, CSV, hidden truth는 0행이고 fit/prediction/upload도 0회다.

## 방법

1. 48개 historical family 전수 원장의 hash를 고정했다.
2. P1/P2/P3 validation audit와 promotion gap matrix의 hash·byte size를 검증했다.
3. P2/P3의 pre-frozen official aggregate receipt를 대조했다.
4. `fresh surface`, `same-Public adaptation`, `closed exact axis`, `new mechanism` 네 gate를 모두 적용했다.
5. 하나라도 통과한 후보가 있을 때만 fit을 허용하도록 했으나 세 문제 모두 통과하지 못해 0-fit으로 종료했다.

Time-series 평가에서는 forecast 문제의 순서를 보존하는 blocked/rolling-origin 설계가 필요하다. [Bergmeir & Benítez 2012](https://doi.org/10.1016/j.ins.2011.12.028) 본 preflight는 노출된 fold를 이름만 바꿔 fresh holdout으로 재사용하지 않았다.

## 이상치 제거에 대한 판정

자동 이상치 제거는 이번 후보에서 금지했다. P1의 이상 구간과 P3의 storm extreme은 제거 대상 noise가 아니라 예측해야 하는 signal일 수 있다. 결과를 본 뒤 extreme을 제거하면 estimand가 바뀌고 또 하나의 adaptive parameter가 생긴다.

허용되는 것은 label·score를 보기 전에 고정한 sensor-QC rule뿐이다. 예를 들어 impossible physical domain, 명시적 sensor failure flag, duplicate key처럼 데이터 계약으로 판정 가능한 항목은 training-only preflight에서 제거할 수 있다. 반면 큰 residual, 낮은 likelihood, 높은 wave height라는 이유만으로 target row를 제거하지 않는다.

## 한계와 강건성

- 이 판정은 broad architecture 전체를 영구 기각하지 않는다. 닫힌 것은 현재 exact recipe와 현재 exposed surface다.
- Public feedback은 official 방향성 증거지만 Private 성능의 독립 확인은 아니다.
- P3의 네 official delta는 서로 다른 contemporary baseline을 포함하므로 절대 성능 ranking chart로 읽으면 안 된다. 각 막대는 해당 후보의 baseline 대비 변화만 나타낸다.
- 새 labels, 새 external forcing target, 또는 완전히 다른 observation support가 생기면 판정을 다시 계산해야 한다.
- 문헌은 방법 선택 원칙을 지지하지만 이 대회의 numeric margin을 보장하지 않는다.

## 다음 단계

1. **P1:** 새 chronological label block이 생기기 전까지 trial18/기존 checkpoint를 보존하고 재학습하지 않는다. 새 block이 생기면 checkpoint rule까지 먼저 hash-freeze한다.
2. **P2:** bin17 champion을 동결한다. 다음 연구는 Public과 무관한 same-season block에서 rank/season factor를 선택하는 DINEOF/low-rank reconstruction으로 제한한다.
3. **P3:** KMA·station-ablation·기존 ERA5 Hs² 축을 닫는다. 최소 3개 fresh, episode-disjoint storm block과 새 wind/wave forcing residual target이 확보될 때만 재개한다.
4. **공통:** 연구 prompt에 `new information source`, `non-duplication`, `fresh confirmation`, `stop if absent`를 필수 필드로 넣는다. 모델 이름만 새롭고 정보면이 같으면 실행하지 않는다.

## 남은 질문

- 대회 종료 후 추가 label block 또는 organizer-side blind evaluation을 확보할 수 있는가?
- P2에서 Public feedback과 독립적인 same-season holdout을 만들 수 있는가?
- P3에서 local station observations와 재분석 wind/wave forcing을 episode-disjoint하게 동기화할 수 있는가?
- P1의 range/event utility를 공식 score와 더 가깝게 정의할 추가 규칙이 있는가?

# P1 공식 component score ledger — 결론

**Lane B 결론은 `ABSTAIN_NO_DEFENSIBLE_0P3_CANDIDATE`다.** 현재 챔피언 `0.833548 / 28.909341점`에서 `+0.3점`을 얻으려면 경험적 환산상 약 `+0.01128748 F1`가 필요하다. 공식 aggregate로 분리된 가장 큰 미세 component인 I 80행은 `+0.285796점`이지만 이미 챔피언에 포함되어 있다. G 15행 `+0.120101점`, router `+0.642207점`, E150 전체 `+0.408627점`, GI2 `+0.007978점`도 모두 이미 포함되어 있어 재조합은 새 후보가 아니다.

S 238행 전체는 두 factorial context에서 표시상 정확히 중립이다. 하지만 이는 내부 유익/유해 행이 없다는 뜻이 아니다. 반대로 어떤 subset을 제거해야 하는지도 알려 주지 않는다. 사전등록 nested S-layer replay(v33c)는 layer를 하나도 선택하지 않았으므로, official score를 행별 감독신호로 사용하는 사후 subset을 만들지 않는 한 materializer를 정당화할 수 없다.

## Family closure와 orthogonal hypothesis

- **폐쇄:** I 제거, G 제거, E150 sparse veto, 이미 제출한 4-row HistGBDT diff, label-shift/EM score transport 재조정, 같은 hash/diff 재제출.
- **미폐쇄 가설:** anchor FN이 장기 이벤트 내부에 집중된다는 메커니즘 자체. 다만 ECDF consensus는 Q4에서 전부 FP였고, v28의 큰 historical improvement는 concentration/slice gate를 실패했으며, v30 label-free guard는 내부 PASS 후 Public에서 `-0.923012점` 역전했다.
- **따라서:** “long-event rescue”라는 연구방향은 남지만 지금 업로드할 후보는 아니다. fresh chronological block 또는 사전봉인 독립 causal covariate 없이 같은 Q2/Q3/Q4를 다시 조정하면 독립 증거가 되지 않는다.

## Expected score

챔피언 유지 점수는 `28.909341`로 확정돼 있다. 새로운 candidate의 방어 가능한 expected-score interval은 없다. S row subset 및 orthogonal rescue의 Public distribution이 aggregate receipts로 식별되지 않기 때문이다. 임의의 넓은 구간을 써서 “0.3점 가능”으로 포장하지 않았다.

## Source coverage

지속 보존된 공식 receipt/manifests에서 2026-08-25 original/A/B, 08-26 router/intersection/union, 08-27 E150 three-way, 08-28 GI probes, 08-29 sparse veto, 08-30 G/S factorial, 08-31 HistGBDT/label-shift, 09-01 v33a를 수집했다. 이 원장은 UI 전체 이력의 완전성을 주장하지 않고, **저장소와 제출 폴더에서 발견 가능한 durable P1 official evidence 전체**를 범위로 한다.

후보 CSV, official test row/value, hidden label, sample/submission 값은 열지 않았고 materialization/upload는 0이다.

# P1 Clean-State CAPA historical falsification — 기술 보고서

## 결론

`p1_clean_state_capa_falsification_20260831_v1`은 기술적으로 정상 완료됐지만 `NO_GO_RESEARCH_ONLY`다. 제출하거나 이 결과를 보고 같은 실험의 penalty를 다시 맞추면 안 된다.

고정 incumbent의 pooled binary row F1은 0.860483604였고 candidate는 0.133098616이었다. ΔF1은 -0.727384987이다. candidate가 추가한 203,574행의 precision은 0.044303300에 불과했다. paired cluster bootstrap 2,000회 CI90은 [-0.779896585, -0.672242860]이고 양수 replicate 비율은 0이었다.

이는 구현 실패가 아니라, 합성 신호에서 통과한 Gaussian-style fixed segment likelihood가 실제 autocorrelated/heavy-tailed residual 표면에서 지나치게 많은 구간을 선택한 과학적 실패다.

## 무엇을 측정했나

- 표면: frozen 2025 Q2/Q3/Q4 historical validation 421,032행
- 비교 기준: full-fraction historical OOF incumbent prediction
- candidate: incumbent와 clean-state CAPA additions의 protected bitwise OR
- 주 지표: pooled binary row micro-F1
- sanity gate: additions precision > incumbent F1 / 2
- 불확실성: positive event와 station×layer×KST normal day를 cluster로 한 paired bootstrap 2,000회
- 금지 범위: official test/sample/submission 값, hidden label, submission CSV, upload 모두 0

## fold별 결과

| Fold | Rows | Incumbent F1 | Candidate F1 | ΔF1 | Additions | Addition precision | Abstained groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q2 | 133,170 | 0.778414 | 0.136654 | -0.641759 | 59,808 | 0.034561 | 5 |
| 2025 Q3 | 176,738 | 0.897059 | 0.141202 | -0.755856 | 80,301 | 0.048991 | 0 |
| 2025 Q4 | 111,124 | 0.909025 | 0.119116 | -0.789908 | 63,465 | 0.047554 | 0 |

세 fold가 모두 같은 방향이므로 특정 분기 하나의 우연으로 설명할 수 없다. Q2에서 prefix support가 없던 신규 I-ORS 5개 layer, 26,062행은 미래 validation 값으로 state를 추정하지 않고 zero-signal abstain했다. Q3/Q4에는 unsupported group이 없었지만 하락폭이 더 컸으므로 Q2 abstention이 주원인도 아니다.

## 고정 모델과 실행 순서

각 fold마다 cutoff 이하 input-only prefix로 robust seasonal clean state를 1회 적합했다. 인접 관측 layer가 있으면 graph residual을 합성했다. decoder는 48/96/192/384/519행의 mean-shift와 linear-drift likelihood를 고정 penalty로 비교했고, isolated point는 삭제하지 않고 winsorize했다. 양의 net gain 후보는 weighted interval scheduling으로 비중첩 집합을 선택했다.

Q2/Q3/Q4의 prediction NPZ와 fold seal을 모두 만든 뒤 `predictions_complete.json`을 기록했다. 그 이후에만 historical `label`과 `anomaly_type`을 열었다. 총 clean-state fit은 3회, supervised fit은 0회다. exactly-once lock은 소비됐고 자동 재시작·결과 기반 retune 권한은 없다.

## 실패 메커니즘

decoder는 Q2/Q3/Q4에서 각각 209/382/240개, 합계 831개 collective segment를 골랐다. additions는 전체 validation의 약 48.35%까지 팽창했다. pooled recall은 0.937154까지 올라갔지만 FP가 194,987개로 늘어 candidate precision이 0.071636까지 무너졌다.

따라서 문제는 "segment family가 아무것도 못 찾았다"가 아니라 "실제 residual에서 null gain의 꼬리가 이론 penalty보다 훨씬 크다"는 것이다. station-layer residual의 autocorrelation, 계절모형 misspecification, scale heterogeneity가 독립 Gaussian likelihood 가정을 깨뜨린 것으로 해석하는 것이 가장 보수적이다. 이 해석은 진단이지 인과 증명은 아니다.

## 독립 QA

`independent-qa.json`은 terminal/result/manifest/attempt-lock hash, manifest 파일 hash, fold별 sealed prediction hash, candidate=incumbent OR additions, incumbent removal 0, fold와 pooled TP/FP/FN/F1, decision, 접근 0을 독립 재계산해 모두 PASS했다. focused pytest는 9개 PASS, Ruff도 PASS다.

## 다음 연구 경계

이 고정 후보는 제출하지 않는다. 같은 실행의 penalty를 결과에 맞춰 조정하지도 않는다.

P1을 다시 열 경우에는 별도 experiment id와 exactly-once 계약으로 다음 순서를 권한다.

1. validation label을 열기 전에 prefix residual block maxima로 empirical null을 만든다.
2. station-layer별 false-alarm ceiling 또는 전체 proposal-share budget을 사전 고정한다.
3. cross-layer coherence를 만족하지 않는 단일-layer 구간을 abstain한다.
4. label-free proposal budget preflight를 통과한 경우에만 Q2–Q4 prediction을 seal한다.
5. 이후에만 historical target을 열어 incumbent union을 검증한다.

이 후속은 현재 결과의 penalty 재튜닝이 아니라, 이번에 입증된 calibration 결함을 직접 겨냥하는 새로운 가족이어야 한다.

## 남은 질문

- prefix-only block maxima가 station-layer별 false-alarm rate를 fold 간 안정적으로 맞출 수 있는가?
- event-duration prior와 cross-layer coherence 중 어느 쪽이 addition precision을 더 크게 올리는가?
- target을 열기 전 proposal share 자체로 무가치한 decoder를 중단시키는 ceiling을 어느 입력-only 통계로 정할 것인가?

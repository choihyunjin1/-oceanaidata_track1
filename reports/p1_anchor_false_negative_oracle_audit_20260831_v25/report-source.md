# P1 v25 설계용 false-negative oracle 정찰 결론

현재 anchor는 historical OOF 양성 16,055행 중 3,299행을 놓쳐 FN rate가 `20.5481%`다. 그러나 단순 결측·gap·잔차 크기·station cell만으로 이 행을 안전하게 추가할 수 있는 근거는 약하다. 가장 강한 구조는 **기존 base/peer/e150 연속 확률의 Q2/Q3 고정밀 신호가 Q4에서 완전히 역전되는 calibration-state shift**다. 따라서 v23의 HGB/top-k, v24의 GCE inner threshold, v19의 run extension과 중복되지 않는 다음 독립축은 `prequential unlabeled label-shift EM odds correction over frozen source probabilities`가 적합하다. 이것은 제안일 뿐이며 이 감사에서는 후보·threshold·fit·lock을 만들지 않았다.

## 전체 및 사건 위치

- rows `421,032`, truth positives `16,055`, anchor FN `3,299`, truth events `141`
- FN의 `3,253/3,299 = 98.61%`가 event interior다. onset은 25, end 14, singleton 7이다.
- onset miss rate `30.49%`, interior `20.55%`, end `17.07%`, singleton `11.86%`
- FN의 `3,278/3,299 = 99.36%`가 길이 25행 이상인 truth event에 속한다.

따라서 새 onset/boundary/run-length extension이 주 병목이라는 증거는 없다. 실제 병목은 이미 오래 지속되는 사건 내부에서 anchor가 중간 구간을 놓치는 것이다. 이 결과는 truth-event oracle geometry라서 직접 배포 feature나 후보 gate로 사용할 수 없다.

## station/layer/quarter oracle 분포

FN support가 큰 cell은 다음과 같다. 괄호는 `FN / anchor-negative`, 즉 cell 전체를 추가했을 때의 marginal precision이다.

- I-ORS layer1 Q2: `551 / 5,989 = 9.20%`
- I-ORS layer1 Q4: `363 / 7,126 = 5.09%`
- S-ORS layer2 Q2: `333 / 9,924 = 3.36%`
- G-ORS layer1 Q3: `329 / 7,046 = 4.67%`
- S-ORS layer6 Q2: `329 / 9,455 = 3.48%`
- I-ORS layer2 Q3: `266 / 12,262 = 2.17%`

모두 incumbent F1/2 약 0.45보다 훨씬 낮다. station/layer/quarter 자체를 router로 쓰는 방식은 안전하지 않다.

## event·gap·missingness

- 10분 이하 정상 cadence: `3,299 / 406,907 = 0.811%`
- 10분 초과 gap의 FN은 전 구간 합계 0
- 완전 관측·gap 없음: FN 3,266, marginal precision `0.817%`
- PSAL만 missing·gap 없음: FN 33, marginal precision `0.522%`
- depth missing 또는 `has_gap_before=1`: FN 0

결측 제거나 gap 복원은 현재 FN 병목의 직접 해법이 아니다. validation/outlier 삭제를 정당화하지도 않는다.

## frozen probability oracle

Anchor-negative 행에서 pooled marginal precision은 확률이 높을 때 컸다.

- base `0.50–0.75`: `106/194 = 54.64%`; `0.75–0.90`: `30/38 = 78.95%`
- peer `0.50–0.75`: `70/93 = 75.27%`; `0.75–0.90`: `23/28 = 82.14%`
- e150 `0.90–0.95`: `125/225 = 55.56%`; `>=0.99`: `750/975 = 76.92%`

하지만 이 pooled 값은 심한 fold reversal을 숨긴다.

| frozen signal | Q2 | Q3 | Q4 |
|---|---:|---:|---:|
| base >=.50 | 73/134 = 54.48% | 72/92 = 78.26% | 0/16 = 0% |
| peer >=.50 | 66/79 = 83.54% | 38/48 = 79.17% | 0/5 = 0% |
| e150 >=.90 | 699/929 = 75.24% | 224/338 = 66.27% | 2/56 = 3.57% |
| e150 >=.99 | 526/623 = 84.43% | 223/304 = 73.36% | 1/48 = 2.08% |
| base>=.50 AND peer>=.50 AND e150>=.90 | 33/34 = 97.06% | 23/23 = 100% | support 0 |

따라서 pooled threshold, stable top-k, 고정 consensus를 이 표에서 뽑아 쓰는 것은 금지한다. 특히 Q4 outer truth를 보고 cutoff를 정하면 명백한 posthoc leakage다.

## causal residual bins

Q2 anchor-negative covariate 분포로만 경계를 고정해 Q3/Q4에도 적용했다.

- `abs(temp median residual 24h)`의 최고 marginal precision은 중상위 bin에서 약 `1.38%`
- `abs(peer residual)`은 Q2 50–75% bin에서 `1.63%`; missing peer residual은 `2.13%`
- `abs(temp robust z 24h)` 전 구간 최고 약 `0.97%`

잔차 크기 단독 선택은 F1 precision floor와 두 자릿수 이상 차이가 난다. outlier 삭제나 fixed residual threshold는 재개 근거가 없다.

## v23/v24와 비중복인 단일 추천축

추천은 **frozen source continuous-probability vector의 prequential label-shift EM prior correction**이다.

1. 각 outer train-prefix의 앞부분에서 base/peer/e150 frozen logits를 낮은 차원의 monotone calibration model로 fit한다.
2. inner tail labels에서만 calibration과 add-only threshold를 결정한다.
3. outer fold에서는 labels를 읽기 전에 frozen score distribution만 이용해 target prevalence를 EM으로 추정하고 posterior odds를 연속적으로 보정한다.
4. hard quarter/station router, HGB, top-k, fixed consensus, GCE score 재사용, event run extension은 쓰지 않는다.
5. Q4 score-mass collapse가 label-free로 관찰돼도 Q4 truth는 threshold·EM stop·feature 선택에 사용하지 않는다.

이 축은 Q2/Q3의 강한 joint probability signal과 Q4 reversal을 동시에 다루는 구조적 가설이다. 다만 prior-shift assumption도 확인된 사실이 아니며, conditional shift이면 실패한다. 새 v25 후보가 된다면 별도 duplicate audit와 result-before-seal config가 필요하고, 이 oracle 표의 숫자로 모델 차원·threshold·iteration을 정해서는 안 된다.

## 접근 및 해석 경계

- historical OOF read 1, candidate 0, model fit 0, threshold search 0
- attempt lock 0, official/hidden/CSV/upload 0
- `result.json`은 datetime-unit 기술 결함으로 event geometry만 무효이며 보존했다.
- authoritative result는 `result.corrected.json` SHA-256 `c63810d5eb1342f8ba7a2b82bb72c818cd2baba7bda7dc58313c9fc96aee4552`다.
- 모든 Q2/Q3/Q4 truth 집계는 exploratory oracle evidence이며 독립 validation이나 후보 선택 증거가 아니다.

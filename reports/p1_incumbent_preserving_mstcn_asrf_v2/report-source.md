# P1 고용량 incumbent-preserving MS-TCN++/ASRF v2

기술 보고서 · 2026-08-27  
팀: 분당독고다이  
상태: **NO_GO_CONFIRMATORY · 실행 및 독립 QA 완료**

## 결론

이번 고용량 전략은 **공식 제출 후보로 기각한다**. Q2에서는 width 512·epoch 125·threshold 0.9가 Router 대비 ΔF1 `+0.098157`로 선택됐지만, 이는 882개 격자의 고립된 낙관적 최대점이었다. 완전히 새로 적합한 확인창에서 Q3는 `+0.013299` 개선됐으나 Q4가 `-0.031484`로 붕괴해 pooled F1은 `0.902917`에서 `0.897777`로 `-0.005140` 하락했다.

추가한 753행의 pooled precision은 `0.381142`였다. incumbent-preserving OR가 기존 F1을 높이기 위한 최소 추가 precision `0.451459`보다 7.03%p 낮다. 특히 Q4 추가 precision은 `0.098930`에 불과했다. 21일 block bootstrap CI90도 `[-0.028100, +0.014067]`로 0을 포함한다. 따라서 연구 성공과 high-impact 공식 probe gate가 모두 실패했으며 CSV 생성·공식 업로드·공식 +3 주장은 모두 0회다.

이 실패는 산출물 손상이나 계산 오류가 아니다. 독립 QA에서 40/40 산출물, 137개 무결성 assertion, Q3·Q4 지표와 10,000회 bootstrap 재계산이 모두 일치했다. 구조적 결론은 **Q2 단일 최대점 선택과 전역 threshold를 통한 add-only 복구가 시기·station 변화에 견디지 못했다**는 것이다.

## 목표와 기준선

공식 최고 P1 결과는 F1 `0.817873`, 문제 점수 `28.492736`이다. 같은 날 세 P1 제출의 점수 차로 계산한 공식 환산 기울기는 F1 1.0당 약 `26.578036점`이다. 따라서 현재 최고에서 3점을 더 얻으려면 F1 약 `+0.112875`, 즉 공식 F1 `0.930748...`가 필요하다. 실행 계약은 보수적으로 `0.930749`를 사용한다.

Q3+Q4 로컬 current-Router 기준선은 다음과 같다.

| TP | FP | FN | F1 | 완전 FN 복구 상한 |
|---:|---:|---:|---:|---:|
| 8,961 | 203 | 1,724 | 0.9029170235 | 0.9905900895 |

로컬 F1 `0.930749`까지 필요한 증가는 약 `+0.027832`다. 이는 계산상 가능하지만 공식 +3의 운송 증명은 아니다. 과거 Router 개선은 로컬 `+0.00222991`에 대해 공식 `+0.024163`으로 방향은 같았으나 크기 비율이 약 10.84배였다. 로컬과 공식의 절대 크기 대응은 안정적으로 보지 않는다.

## 전략 가설

현재 Router의 모든 양성은 그대로 유지한다. 새 모델은 2,048행 시간창에서 장기 사건 확률, 시작·종료 경계, 사건 유형을 함께 예측한다. 최종 후보는 아래처럼 단조 추가만 허용한다.

`candidate = exact current Router OR decoded long-event proposal`

따라서 기존 양성의 `1→0` 전이는 0이다. 개선 여부는 새로 추가한 행이 Router가 놓친 FN을 충분한 precision으로 복구하는지에 달려 있다.

## 모델과 계산 예산

공통 구조는 MS-TCN++에서 영감을 받은 dual-dilated prediction generator, 3개 local refinement stage, ASRF에서 영감을 받은 시작·종료 boundary head와 5개 anomaly-type head다. 입력은 fail-closed 감사를 통과한 165개 특징이다.

| 용량 | Batch | 학습 파라미터 | 측정 optimizer step | 측정 peak VRAM |
|---|---:|---:|---:|---:|
| width 256 | 128 | 13,177,099 | 0.3360초 | 17.65 GiB |
| width 512 | 64 | 52,568,587 | 0.4705초 | 18.08 GiB |

두 용량을 동시에 GPU에 올리지 않는다. Q2는 약 2.0–2.7시간, 전체 terminal 실행은 선택된 epoch와 width에 따라 약 3.7–8.3시간으로 예상한다. 300 epoch는 공식 구현을 재현하는 값이 아니라 수렴·과적합 곡선을 관찰하기 위한 사전 고정 상한이다.

## 누수 방지와 특징 계약

전체 시계열 또는 미래 run 길이에 의존하는 다음 캐시 특징은 projection 단계에서 제거한다.

- `nominal_depth_m`
- `depth_regime`
- `plateau_full_length`
- `plateau_count`

depth regime은 현재 행 `depth_raw`와 각 학습 prefix에서만 적합한 threshold로 다시 계산한다. bounded centered feature의 최대 미래 support는 168시간, validation feature의 최대 과거 support는 169시간이다. 합계 337시간보다 긴 21일 10분 간격 purge를 적용하며 실제 최소 분리는 504.17시간이다. 특징 allowlist는 캐시 metadata 80개와 정확히 일치해야 하며 미분류 열은 자동 허용하지 않는다.

## 선택과 확인 절차

1. Q2 이전 prefix만 사용해 width `256/512` × seed `3개`를 각각 300 epoch 완주한다.
2. epoch `1, 2, 3`과 이후 5 epoch 간격의 총 63개 checkpoint에서 Q2 blind probability를 생성한다.
3. 각 width·epoch에서 3-seed 평균을 만들고 high threshold `0.3–0.9`, low threshold `high/2`의 모든 후보를 Q2 truth 접근 전에 저장·해시 봉인한다.
4. Q2 truth를 한 번 열어 `Router OR proposal` F1로 width·epoch·threshold를 고른다. best seed 단독 선택은 금지한다.
5. 선택 사양을 고정하고 Q3와 Q4 prefix에서 각각 3개 seed를 새로 학습한다. 두 fold의 blind prediction을 모두 봉인한 뒤에만 두 truth를 연다.
6. Q3+Q4 pooled 지표와 fold·station별 부호, 두 fold를 합친 전역 KST 날짜 단위의 21일 paired circular block bootstrap 10,000회를 계산한다. 같은 날짜의 두 fold 행은 하나의 cross-section으로 함께 재표집한다.

Q3·Q4는 전역 달력 구간으로 완전히 분리된 fold가 아니다. 두 fold의 전역 envelope는 2025-10-01 하루에 19시간 50분 겹치지만, 두 fold가 모두 존재하는 16개 `(station, year, layer)` 시계열에서 Q3의 마지막 행과 Q4의 첫 행 사이 최소 간격은 10분이고 non-positive gap은 0이며 ordered row-key overlap도 0이다. 따라서 이를 `series-local chronological/key-disjoint frozen folds`로 명시한다. Q3 truth와 지표는 Q4 blind prediction까지 모두 봉인된 뒤에만 연다.

## 사전 고정 gate

### Q2 연구 지속

- Router-union ΔF1 ≥ 0.010
- 추가 행 precision ≥ 0.70
- eligible long-event Router-FN 회복률 ≥ 0.05
- 후보 FP / Router FP ≤ 2.0
- Router positive 제거 = 0

### Q3+Q4 연구 성공

- pooled ΔF1 ≥ 0.015
- Q3, Q4 모두 ΔF1 > 0
- paired block-bootstrap CI90 lower > 0
- Router positive 제거 = 0

### 공식 probe 검토 가능

- pooled F1 ≥ 0.930749
- pooled ΔF1 ≥ 0.027832
- Q3, Q4 각각 ΔF1 ≥ 0.010
- 추가 행 precision ≥ 0.75
- 3개 station 중 최소 2개 개선
- CI90 lower ≥ 0.015
- Router positive 제거 = 0

이 gate의 명칭은 `OFFICIAL_PROBE_ELIGIBLE`이다. 공식 +3을 보장하거나 확인했다는 표현을 금지한다.

## 실행 결과

정식 launcher-only 단발 실행은 2026-08-27 KST 05:47:41부터 10:03:14까지 `15,333.13초` 수행됐다. 재시도는 없었고 Q2 6개 fit은 각 300 epoch, Q3·Q4 6개 refit은 각 125 epoch를 모두 완료했다. 12개 이력의 nonfinite 합은 0이다.

### Q2 선택 전용 결과

선택 사양은 width `512`, batch `64`, epoch `125`, high threshold `0.9`, 3-seed raw probability 평균, 최소 추가 구간 19행, boundary snap 12행이다.

| 기준 | Router | 후보 | 차이/보조 지표 |
|---|---:|---:|---:|
| F1 | 0.792275574 | 0.890432823 | +0.098157249 |
| 추가 행 precision | — | — | 0.884220 |
| long-event recall gain | — | — | +0.192401 |
| FP 비율 | — | — | 1.298795× |
| Router 양성 제거 | — | — | 0행 |

width 512의 Q2 최선 ΔF1은 width 256 최선보다 `+0.01849` 높아 큰 용량을 시험한 선택 자체는 가치가 있었다. 그러나 epoch 125는 인접 epoch 120의 `+0.071073`, 130의 `+0.078542`보다 유난히 높은 `+0.098157`의 고립 peak였고, 882개 후보 중 유일한 최대점이다. 따라서 Q2는 사양 선택 증거일 뿐 승격 증거가 아니다.

### Q3·Q4 확인 결과

| 구간 | Router TP/FP/FN | 후보 TP/FP/FN | Router F1 | 후보 F1 | ΔF1 | 추가 precision |
|---|---|---|---:|---:|---:|---:|
| Q3 | 5,241 / 74 / 1,156 | 5,491 / 203 / 906 | 0.894980 | 0.908279 | +0.013299 | 0.659631 |
| Q4 | 3,720 / 129 / 568 | 3,757 / 466 / 531 | 0.914342 | 0.882857 | -0.031484 | 0.098930 |
| Pooled | 8,961 / 203 / 1,724 | 9,248 / 669 / 1,437 | 0.902917 | 0.897777 | -0.005140 | 0.381142 |

OR 후보는 Router 양성 0행을 제거하지 않았고 FN 287행을 복구했지만 FP 466행을 추가했다. pooled 추가 precision `0.381142`는 F1 개선에 필요한 Router F1/2 `0.451459`보다 낮다. Q4에서는 필요한 `0.457171` 대비 실제 `0.098930`으로, 성능 붕괴의 주된 원인은 Q4 false-positive 폭증이다.

| Station | Router F1 | 후보 F1 | ΔF1 |
|---|---:|---:|---:|
| G-ORS | 0.800962 | 0.831093 | +0.030131 |
| I-ORS | 0.874434 | 0.870084 | -0.004350 |
| S-ORS | 0.935037 | 0.923869 | -0.011168 |

개선 station은 1/3이다. 163개 연속 KST 날짜를 대상으로 한 21일 paired circular moving-block bootstrap 10,000회에서 평균 ΔF1은 `-0.005624`, CI90은 `[-0.028100, +0.014067]`, 양수 replicate 비율은 `35.15%`였다. Q3·Q4 공유 날짜 1일은 pooled cross-section으로 한 번만 표집했다.

### Gate와 수렴 판정

- 연구 성공: **FAIL** — pooled ΔF1, Q4 부호, CI90 하한 실패.
- high-impact 공식 probe: **FAIL** — pooled endpoint `0.897777 < 0.930749`, ΔF1 `-0.005140 < +0.027832`, 추가 precision `0.3811 < 0.75`, CI90 하한 `-0.02810 < 0.015`, 개선 station `1/3 < 2/3`.
- 공식 제출 후보: **아님**. submission 생성 0회, upload 0회, 공식 +3 주장 false.

Q3 refit의 final/min loss 배수는 `5.87×, 1.15×, 11.11×`, Q4는 `12.60×, 1.00×, 26.54×`였다. 모든 값은 유한했지만 선택 epoch 125의 최종 learning rate가 약 `1.99e-4`여서 여러 시드에 후반 최적화 상태 전이가 남았다. 따라서 “125 epoch에서 수렴했다”는 주장은 할 수 없다.

### 독립 QA

- 과학·통계 QA: `P0=0 / P1=0 / P2=1`; 지표·bootstrap·gate 재계산 일치. P2는 유한하지만 큰 tail-loss 불안정성이다.
- 무결성 QA: `PASS_CONTROLLED`, `P0=0 / P1=0 / P2=2`; exact inventory 40/40, 독립 assertion 137개 PASS. P2 두 건은 적대적 동일 프로세스/동시 경로 교체라는 기존 controlled-host 신뢰 경계다.
- terminal SHA-256: `7640cc0e29f364a26cd8199a7e9a55acdf329699cd5923679d8f0d513c4af2b1`
- confirmatory metrics SHA-256: `964cae7d7dbb9f413244462eb9258e883e14f6073ad5549fadf482cb9cd03bd4`
- execution seal SHA-256: `2d42ce76966876f33daf0bd3e8e62051876f95f92e866588713bcfb84886bb25`

### 다음 전략 목표

동일한 전역 threshold·단일 Q2 최대점 전략은 중단한다. 다음 후보는 아래 세 결함을 동시에 겨냥해야 한다.

1. **다중 창 worst-fold 선택**: 사전 고정한 최소 3개 historical development window에서 각 후보를 모두 평가하고, 평균이나 pooled 최고점이 아니라 `min(window ΔF1)`을 1차 목적함수로 삼는다. 모든 창에서 추가 precision이 적어도 해당 창 Router F1/2를 넘어야 하며, pooled F1은 동률 해소에만 사용한다. 이 규칙은 Q2 epoch 125와 같은 고립 peak가 다른 창의 손실을 가리고 선택되는 것을 막는다.
2. **precision-first abstention**: add-only proposal을 즉시 OR하지 않는다. 과거 자료만으로 적합한 station·regime별 calibration과 causal support 조건을 통과한 구간만 추가하고, 표본이 부족하거나 precision 하한을 입증하지 못한 구간에서는 0개를 추가하는 abstention을 허용한다. 개발 단계의 최소 조건은 창별 추가 precision이 각 Router F1/2를 넘고, 3개 station 중 최소 2개에서 ΔF1가 양수인 것이다. 공식 probe 검토에는 기존의 더 강한 pooled 추가 precision `0.75` 기준을 그대로 유지한다.
3. **optimizer 안정화**: 구조와 threshold 탐색을 다시 늘리기 전에 사전 고정한 더 낮은 learning-rate tail과 checkpoint/seed averaging을 비교한다. 선택 epoch의 단일 raw checkpoint가 아니라 동일한 다중 창 worst-fold 기준으로 안정화 방법 하나를 고르며, nonfinite 0뿐 아니라 tail-loss max/min, final/min, 창별 예측 변동도 함께 gate한다.
4. **새 outer 확인창 보존**: 이번 실행에서 Q2·Q3·Q4 결과가 모두 공개됐으므로 이후 이 창들은 진단·개발 자료일 뿐 새 승격 증거가 될 수 없다. 다음 승격 판정에는 아직 열지 않은 시간 구간을 outer holdout으로 남기거나, outer window가 매번 한 번만 열리는 nested forward-chaining 절차를 사용한다.

다음 실험의 핵심 질문은 “더 큰 모델인가”가 아니라 **Q4와 같은 분포 변화에서도 추가 FP를 억제하면서 Router FN만 복구할 수 있는가**다.

다음 실행의 승격 순서는 세 단계로 고정한다.

1. **개발 지속**: 모든 development window에서 ΔF1 > 0, 추가 precision > 해당 Router F1/2, tail-loss 안정성 gate 통과.
2. **연구 성공**: 미사용 outer holdout에서 pooled ΔF1 ≥ 0.015, 모든 outer fold 양수, CI90 lower > 0, Router 양성 제거 0.
3. **공식 probe 검토**: 기존 high-impact 기준 전체 통과 후에만 별도 사용자 승인을 요청한다. 로컬 통과만으로 공식 +3을 주장하지 않는다.

## 다음 실험에서 먼저 답할 질문

- Q4의 FP 337행은 특정 station·regime·계절 구간에 집중됐는가, 아니면 전역 calibration 이동인가? 이 분석은 다음 abstention 규칙의 가설 생성에만 쓰고 Q3·Q4를 다시 확인창으로 재사용하지 않는다.
- tail-loss 급등은 learning-rate schedule, temporal-smoothing 항, 희소 양성 batch 구성 중 무엇에 가장 민감한가? 한 번에 하나의 요인만 바꾸는 사전 고정 ablation으로 구분한다.
- 다중 창에서 필요한 추가 precision 하한을 유지하면서도 공식 +3 목표에 필요한 recall 증가를 확보할 수 있는가? abstention이 안정성만 높이고 복구량을 지나치게 줄이는지도 함께 확인한다.
- 완전히 미사용인 outer 확인 구간이 충분한가? 없다면 새 공식 제출 전에 검증 불확실성이 더 크다는 점을 명시하고, 공식 기회는 구조 검증용 probe로만 해석한다.

## 한계

- MS-TCN++와 ASRF의 직접 실증 대상은 해양 QC가 아니라 비디오 동작 분할이다.
- width 256/512와 300 epoch는 공식 MS-TCN++ 기본 설정보다 큰 외삽이다.
- 3개 seed는 최소 안정성 점검이며 충분한 seed 불확실성 추정이 아니다.
- Q3·Q4는 과거 분석에서 기준 점수가 알려진 retrospective window다. 이번 후보에 대해서만 prediction-before-truth 절차를 지킨다.
- Q3·Q4의 전역 달력 envelope는 완전히 분리되지 않는다. 검증 독립성은 전역 분기 이름이 아니라 시계열별 엄격한 순서, exact-key disjointness, 두 fold를 함께 묶는 날짜 bootstrap에 의존한다.
- 21일 block bootstrap은 시간 표본 변동을 다루지만 Q2 다중선택과 모든 HPO 불확실성을 포함하지 않는다.
- official score transport는 관측점이 적고 과거 local/official 효과 크기가 크게 달랐다.

## 근거와 재현성

문헌의 직접 주장, 본 실험의 전이 추론과 적용 한계는 [주장·근거 원장](./claim-source-ledger.md)에 분리해 기록했다. 정량 결론의 기준 산출물은 `terminal_result.json`, `confirmatory_metrics.json`, `q2_selection.json`, 두 confirmatory blind receipt와 여섯 refit history다. 보고서의 주요 지표는 이 산출물에서 독립 재계산했으며, 위 SHA-256과 exact inventory 검증을 통과했다.

# Canonical report source — conditional transport Deep Research v16

작성일: 2026-08-28 KST
상태: P1/P2/P3 사전등록 단발 실행 및 독립 QA 대기
공식 접근·CSV·업로드: 0

## 연구 질문

직전 사이클의 유망하지만 운반성이 불안정한 신호를 결과 기반 튜닝 없이, training-only support와 독립 source gate로 안전하게 승격할 수 있는가?

## 사전등록 후보

### P1

- ID: `p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1`
- 변경 대상: I/S only, G exact no-op.
- 구조: 비동기 peer 관측을 ±10분으로 정렬하고 leave-one-layer-out latent-state score를 구축.
- 고정 kernel memory: 6h/48h.
- calibration: 84일 block conformal, alpha 0.01.
- 보호: e150 anchor union, deletion 금지.
- support gate: cell별 양기간 peer coverage 0.95 이상, matched 500 이상, 정상 calibration block 100 이상.

결과:

- scientific execution 1, learned-model fit 0.
- Q2/Q3/Q4 정상 calibration block 19/74/126.
- peer coverage를 포함한 전체 조건 통과 cell 2/15, 8/15, 14/15.
- `NO_GO_SUPPORT_EXACT_E150_NO_OP`.
- sealed prediction 421,032 rows, anchor와 union SHA가 동일.
- outer truth 0, official 0, submission 0.

해석: asynchronous GP family 일반이 반증된 것이 아니라 이 exact calibration granularity가 식별 불가. threshold 완화는 새 계약 없이는 금지.

### P2

- ID: `p2_alpha50_supervised_rank1_trainonly_regime_veto_20260828_v1`
- 고정 base correction: 직전 α50 supervised rank-1 결과를 bit-exact 사용.
- veto unit: KST 14-day bin.
- eligibility: outer-training 내부 LOBO에서 source blocks>=2, profiles>=100, KST days>=10, 5,000 day-bootstrap CI90 upper<0.
- leakage guard: held block label과 fit block 자기 label을 α50 reference에서 제외.

결과:

- 첫 outer fold는 May-Jun, Jul-Aug의 두 training block만 존재.
- held H를 숨기고 fit F의 자기 label까지 제외하면 season bin 13의 complete prefix seasonal OAS row가 0.
- prediction commitment 0, outer truth 0, prediction file 0, retry 0.
- `NO_GO_IMPLEMENTATION_PREFLIGHT`.

해석: 예외를 무시하거나 reference를 in-sample로 만들면 실험은 돌아가지만 누수 계약이 바뀐다. 따라서 같은 experiment ID로는 terminal.

### P3

- ID: `p3_era5_wave_directional_energy_memory_20260828_v1`
- base/enriched: 동일 CatBoost·seed·rows, 286 vs 306 features.
- 신규 20 features: Hs²-weighted circular concentration 6/12/24/48h, relative cos/sin, 24h signed/absolute turn, 10 masks.
- split: 2014–20 train 7,311 cases, 2021–23 held 492 cases.
- prediction: base + 0.20*(enriched-base) only at 18/24h.
- gate: pooled Δ<0, CI90 upper<0, three years nondegrade, coverage, max slice cap.
- fresh shadow: 2024-02-01–2024-05-01, 58 cases G/I/S 22/19/17; source gate 통과 때만 truth open.

결과:

- source base/candidate RMSE 0.545995677/0.546183396m; Δ +0.000187718m.
- bootstrap CI90 [-0.000245637,+0.000628179].
- year Δ: 2021 -0.000184038, 2022 +0.001148649, 2023 -0.000261055m.
- station Δ: G +0.000263506, I +0.000082249, S +0.000263669m.
- 18h +0.000600561, 24h +0.000260142m.
- `NO_GO_SOURCE_GATE`; shadow truth 0, outer truth 0, official 0.

해석: 현재 관측값으로 만든 mean-direction memory proxy는 source에서 독립 증분 정보를 제공하지 못했다. 실제 directional spread/partition이 없으면 같은 family를 확장할 근거가 약하다.

## 비교 결론

| 문제 | 데이터 지지 | 모델 fit | 검증 결과 | 공식 후보 |
|---|---:|---:|---|---|
| P1 | 실패 | 0 | exact no-op | 아니오 |
| P2 | 구조적으로 식별 불가 | 0 new outer model | terminal preflight | 아니오 |
| P3 | 통과 | 2 | source 악화 | 아니오 |

이번 결과의 공통 메시지는 모델 용량 부족이 아니다. P1/P2는 정직한 평가를 위한 표본·분할 구조가 부족하고, P3는 새 proxy의 독립 정보성이 source에서 반증됐다. 더 큰 모델이나 더 긴 학습은 이 세 실패 원인을 직접 해결하지 않는다.

## 후속 연구 규칙

1. P2는 세 개 이상 독립 prefix block 또는 label-free base reference를 확보하기 전 같은 veto를 재개하지 않는다.
2. P1은 cell별 threshold 완화가 아니라 station-level partial pooling을 새 contract로만 검토한다.
3. P3는 actual directional spread·partition source가 없으면 mean-direction proxy family를 닫는다.
4. official submission은 새 candidate가 local/source gate를 통과한 후 별도 사용자 승인으로만 진행한다.

## 근거 문헌

- https://proceedings.mlr.press/v70/futoma17a.html
- https://proceedings.mlr.press/v84/herlands18a.html
- https://proceedings.mlr.press/v75/chernozhukov18a.html
- https://proceedings.mlr.press/v238/li24g.html
- https://www.ecmwf.int/en/elibrary/81373-ifs-documentation-cy48r1-part-vii-ecmwf-wave-model
- https://confluence.ecmwf.int/spaces/FUG/pages/673550584/Section%2B2A.3.1%2BWave%2Bmeasures%2Band%2Bdefinitions
- https://journals.ametsoc.org/abstract/journals/phoc/18/7/1520-0485_1988_018_1020_amftra_2_0_co_2.xml

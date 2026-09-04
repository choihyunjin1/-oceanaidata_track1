# 2026-08-31 parallel public-transport repair cycle

Audience: model-selection and submission operator
Status: `ALL_PROBLEMS_PASS_SUBMISSION_READY_NOT_UPLOADED`
Scope: P1 v30, P2 v7, and P3 v19 internally promoted candidates and their materialized submission files.

## 결론

- 세 문제 모두 최소 한 개의 내부 채점 PASS 제출본을 확보했다. 보정 예상 점수는 P1 `+0.043013217점`, P2 `+0.013765176점`, P3 `+0.018504580점`으로 모두 inclusive `+0.01점` gate를 통과했다.
- P1 v30은 2개 prequential historical fit에서 33개 추가가 모두 TP였고 pooled ΔF1 `+0.001820930`, Q3 `+0.003104783`, Q4 `0`, bootstrap CI90 `[+0.000764423, +0.003172068]`이었다. 공식 169,011행 CSV도 구조 QA를 통과했다.
- P2 v7은 내부 PASS이며 제출 CSV가 준비됐다. 보수적인 bootstrap 상단 기준 raw 기대 개선은 `+0.135447268점`, 고정 transport penalty `0.121682092점` 차감 후 `+0.013765176점`으로 inclusive `+0.01점` gate를 통과했다.
- P3 v19도 내부 PASS이며 제출 CSV가 준비됐다. 중앙 pooled 개선을 점수로 바꾼 raw 기대 개선은 `+0.068090634점`, 고정 family penalty `0.049586054점` 차감 후 `+0.018504580점`으로 inclusive `+0.01점` gate를 통과했다.
- 2026-08-31 21:22~21:23 KST에 승인된 P1/P3 CSV를 업로드했다. P1은 Public F1 `0.798819`, `27.986329점`으로 기준 `28.909341점`보다 `-0.923012점` 하락했다. P3는 RMSE `0.589840`, `23.971758점`으로 직전 비교값보다 `+0.017717점` 상승했으나 개인 최고 `24.203599점`에는 못 미쳤다.
- P2는 페이지가 오늘 남은 제출 `0/3`과 2026-09-01 00:00 KST 초기화를 표시해 파일 선택·제출 버튼이 비활성화되어 업로드하지 못했다. 제출 삭제나 우회는 수행하지 않았다.
- Hidden truth 접근은 0이다. P1의 official label-shift EM target prevalence가 `0.999999` 경계에 닿았던 경고는 실제 Public 급락으로 확인됐고, 해당 family의 내부→Public transport 가정은 반증됐다.

## Official submission result

| Problem | Time (KST) | Public result | Comparison | Verdict |
|---|---|---:|---:|---|
| P1 v30 | 2026-08-31 21:22 | F1 `0.798819`, `27.986329점` | previous best `28.909341점`; Δ `-0.923012점` | regression; not champion |
| P2 v7 | not uploaded | quota `0/3` | reset displayed at 2026-09-01 00:00 | blocked by daily limit |
| P3 v19 | 2026-08-31 21:23 | RMSE `0.589840`, `23.971758점` | immediate comparator `23.954041점`; Δ `+0.017717점` | transport estimate matched closely, but not personal best |

Machine-readable receipt: `reports/parallel_public_transport_repair_cycle_20260831_v1/official-submission-receipt.json`.

## Candidate registry

| Problem | Candidate | Internal evidence | Transport-adjusted expectation | CSV | QA |
|---|---|---|---:|---|---|
| P1 | `P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM` | pooled ΔF1 `+0.001820930`; Q3 `+0.003104783`; Q4 `0`; CI90 low `+0.000764423` | `+0.043013217점` | 169,011 rows, SHA `639c26cd…efa3` | strict internal PASS; postrun QA PASS; validator PASS |
| P2 | `P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE` | October ΔRMSE `-0.013071643`; bootstrap CI90 high `-0.010794735` | `+0.013765176점` | 26,061 rows, SHA `c6f2a7e0…620` | independent PASS; root-ready PASS; final cross-QA PASS |
| P3 | `P3_1_KMA_CONTINUOUS_WAVE_POWER_FACTOR` | pooled ΔRMSE `-0.004290325`; 4/6 blocks; both CI90 uppers below 0 | `+0.018504580점` | 1,200 rows, SHA `b1b72f90…d0c4` | independent 15/15 PASS; delivery PASS; final cross-QA PASS |

## P1 v30

- Result: `COMPLETE_INTERNAL_ONLY` with strict internal PASS; nested fits 2; runtime `22.792s`.
- Frozen method: base/peer/e150 three-logit calibrator, train-prefix threshold, label-free target-prior EM, station-layer score-discrepancy reliability bound, add-only decoder, and per-KST-day 0.5% cap.
- Internal evidence: 33 additions, 33 TP, 0 FP, 0 removals; pooled ΔF1 `+0.001820930400542653`; Q3 `+0.003104782991018129`; Q4 `0`; bootstrap probability improved `1.0`; CI90 `[0.0007644228497011463, 0.0031720682353547362]`.
- Recomputed transport: raw `+0.04839690827670436점`, P1-specific v3 penalty `0.005383691373120247점`, calibrated `+0.04301321690358412점`.
- CSV: `C:\Users\cedis\PycharmProjects\PythonProject\submissions\p1_public_transport_repair_cycle_20260831_v30\P1_submission.csv`.
- Structural QA: 169,011 rows; columns `station,year,layer,time,label`; official keys and order exact; duplicate keys/rows 0; binary finite integer labels; official validator PASS.
- Hashes: internal result `5f47835934bdfababb2a86c7674596a39ddc64da677609c7992d83c359b04826`; materialization result `3bf4080ca09315cb2bfb076e43f902c6795cfc6d2b4651d22c2b8784b7443a75`; CSV `639c26cda576da74880e3f887b8653465db16c8572deedf225e4ae39ae51efa3`; postrun QA `1e7c7f5d9112b227f183b1bfe0875cd50e6afdb744a5195b2d1d9b1086227090`.
- Deployment warning: full-history EM converged to target prevalence `0.999999`, producing 133,153 eligible proposals before the frozen day cap and 755 final additions. This is a large train-to-official score-distribution shift. The policy was not retuned or retried after seeing it.
- Access: official reads occurred only after internal PASS for materialization/QA; hidden truth 0; uploads 0.

## P2 v7

- Result: `COMPLETE_WITH_PASS`; internal fits 2, full fit 1, total 3.
- Passing candidate: ExtraTrees public-benefit gate. RandomForest alternative failed the calibrated point gate and was not materialized.
- Gate statistic: `max(0, -bootstrap_CI90_high_delta_RMSE × 12.5475311)`.
- Recomputed: raw `0.1354472678596296`, penalty `0.12168209161000616`, calibrated `0.013765176249623437`.
- CSV: `C:\Users\cedis\PycharmProjects\PythonProject\artifacts\p2_public_feature_benefit_gate_cycle_20260831_v7\submission\P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE\P2_submission.csv`.
- Structural QA: 26,061 rows; columns `station, layer, time, temp`; official key values/order exact; duplicate keys 0; finite predictions.
- Hashes: result `db35e824d8d4957dac35a281b5d48bb03716c7b920ed94e5e5e5ece8a6ad5a10`; CSV `c6f2a7e02ff3e5064ec653af0a52b117cbf8ae49d80e651a2a96276190f4f620`.
- Access: official rows before internal gate 0; materialization test-index read 26,061; root-ready QA read 26,061; final cross-QA read 52,122; known cumulative test-index reads 104,244; hidden truth 0; uploads 0.

## P3 v19

- Result: `PASS_MATERIALIZED_NOT_UPLOADED`; 6 prefix ECDF calibrations, 0 model fits, 1 full deployment ECDF fit.
- Proxy and policy were sealed before scoring: `wave_power = hs_current² × tp_current`; `alpha18=.20`; `alpha24=.20+.40×prefix_ECDF(wave_power)`; invalid proxy and short leads are comparator no-ops.
- Internal evidence: pooled ΔRMSE `-0.004290325362721004`; 4/6 improved bimonth blocks; episode CI90 `[-0.007585174, -0.000999684]`; block-by-station CI90 `[-0.008060051, -0.000524025]`; worst station-by-lead ΔRMSE `+0.007404054`; changed share exactly `1/3`.
- Recomputed: raw `0.06809063425841472`, penalty `0.04958605409228893`, calibrated `0.018504580166125786`.
- CSV: `C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_KMA_CONTINUOUS_WAVE_POWER_V19\P3_submission.csv`.
- Structural QA: 1,200 rows; columns `case_id, station, lead_h, hs_pred`; official key values/order exact; duplicate keys 0; finite predictions.
- Hashes: result `fba999a9a42878a564ee3f0680604368274e2bcc4b17f9f54b5401842f3cae5c`; CSV `b1b72f905e36df994f82ef8dc5c425328c5e0b0b4d16ec2a0dbfd8497c48d0c4`.
- Access: official rows before internal gate 0; materialization test-index/context/features/components reads `1200/200/200/3600`; delivery QA and final cross-QA each read 1,200 test-index rows; known cumulative test-index reads 3,600; hidden truth 0; uploads 0.

## Why the validation is conservative

Repeatedly selecting against a finite validation surface can overfit the selection criterion itself, so the report separates internal promotion from claims of independent confirmation ([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). Structured temporal/group data can make random cross-validation optimistic, which motivates blocked or purged evaluation and explicit group diagnostics ([Roberts et al., 2017](https://doi.org/10.1111/ecog.02881); [Bergmeir, Hyndman & Koo, 2018](https://doi.org/10.1016/j.csda.2017.11.003)). Cross-validation estimates and conventional uncertainty intervals have nontrivial targets and dependence, so the empirical intervals here are gates rather than guarantees of official performance ([Bates, Hastie & Tibshirani, 2024](https://doi.org/10.1080/01621459.2023.2197686)).

Distribution shift remains the central risk. Weighted conformal guarantees under covariate shift require conditions and density-ratio information not established here; the fixed penalties are empirical governance devices, not conformal guarantees ([Tibshirani et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)). DomainBed shows that model-selection protocol materially affects out-of-distribution comparisons, while group DRO work shows that average performance can conceal poor group behavior and that regularization matters ([Gulrajani & Lopez-Paz, 2021](https://openreview.net/forum?id=lQdXeXDoWtI); [Sagawa et al., 2020](https://openreview.net/forum?id=ryxGuJrFvS)). These sources motivate the protocol; they do not validate the numeric penalties or candidates.

## Limitations and next action

- The historical surfaces have been reused adaptively. PASS means registry and submission readiness, not independent scientific confirmation.
- P1 has the strongest expected gain but also the strongest visible transport warning because official EM prevalence hit its upper boundary. P2 and P3 have smaller calibrated margins over `+0.01점`.
- The prospective P1 gate-v4 audit downgrades max-day and every-slice constraints to warnings only for future candidates; it does not retroactively change v28. P1 v30 did not need that relaxation because it passed the original strict gates.
- P1 and P3 uploads are scored and immutable in the submission ledger. P2 remains locally ready but was not transmitted because the site enforced the exhausted daily quota.

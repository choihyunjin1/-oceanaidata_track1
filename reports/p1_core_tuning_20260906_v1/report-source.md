# P1 core tuning — new retrospective CPU2 contract

**완료: 트리 단독 전체 F1은 +0.001631434 개선되어 연구 후보를 보존한다. 하지만 같은 MS를 결합한 실제 제출 구성의 Q3/Q4 평균은 감소했으므로 이번 사이클에서는 full 13-fit/새 제출본으로 진행하지 않고 기존 b2f17 및 fallback을 유지한다.** 특정 분기 하락이나 P<0.8이 중단 사유가 아니라, [고정 MS bridge](../p1_core_tuning_ms_bridge_20260906_v1/report-source.md)의 동일 구성 pooled 평균 미개선이 판단 근거다.

학습 42/48 actual fits, 1,641.191초; CPU2/GPU0. 새 PID 재생 36/36 exact PASS(223.797초), 독립 QA 158/158 PASS(97.250초). 모델 재학습·공식 입력·CSV·upload·Git 추가 실행 0. 원 시도 및 기존 자산은 불변이다.

## 실측 결과

| 평가 범위 | 행 수 | 새 CPU2 기준 union F1 | inner-selected 후보 F1 | ΔF1 |
|---|---:|---:|---:|---:|
| **사전 주평가 Q2/Q3/Q4 pooled** | 421,032 | 0.841633764 | 0.843265197 | **+0.001631434** |
| 기존 진단 주평가 Q3/Q4 pooled | 287,862 | 0.886019090 | 0.886034323 | +0.000015233 |
| Q2 | 133,170 | 0.759196056 | 0.764561271 | +0.005365215 |
| Q3 | 176,738 | 0.869649123 | 0.870456337 | +0.000807214 |
| Q4 | 111,124 | 0.908802344 | 0.908141703 | −0.000660641 |

전체 primary: 양성 16,055행, TP 12,683→12,665(−18), FP 1,401→1,318(−83), FN 3,372→3,390(+18). 개선은 전체 recall 향상이 아니라 FP 감소와 recall 손실의 교환이다. paired calendar-day 254 clusters, CI90 [−0.003067240, +0.006128381], P(Δ>0)=0.7405. Q3/Q4만의 CI90 [−0.006446543, +0.005569863], P=0.5325. 모두 과거 노출기간의 기술적 부트스트랩이며 새 검정·공식 향상 확률이 아니다.

세 fold 모두 **O_slow(lr0.02, 1400trees)+원 B3**를 해당 inner에서 선택했다. O threshold/B threshold는 Q2 0.15/0.10, Q3 0.60/0.40, Q4 0.20/0.15로, outer 값으로 다시 선택하지 않았다. 기준 24 + 대안 inner 15 + 선택 outer O_slow 3 =42 실제 모델이다. O_regular/B_regular은 inner 평가에서 선택되지 않아 그 정책의 outer 성능을 측정했다고 주장하지 않는다.

Q2 미지원 station-layer 26,062행(양성1,554)은 Δ−0.005720401, 지원 107,108행은 +0.009709988이다. 이 행들을 삭제하지 않았다. Q3/Q4 미지원 행은0이나, 배포 미지원 **연도**의 depth fallback 문제와는 다르다. Worst station-layer I-ORS5: 10,190행/양성407, Δ−0.105943060(TP245→194); worst day 2025-11-10: 2,160행/양성86, Δ−0.735294118(TP50→0). 위험은 별도 보존하며 사후 hard gate로 바꾸지 않았다.

## 완결 증거

- [terminal result](../../artifacts/p1_core_tuning_20260906_v1/terminal_result.json), SHA `1e663dd7d28843be345fff36ecc2d0f09480b28d5014a03eb4a6df3d09678a46`.
- [fresh replay](../../artifacts/p1_core_tuning_20260906_v1/fresh-replay-qa.json), SHA `6362668e645f28a3900bf386d1edc32e71ea13fb8d07a649c510f4e377ccd48c`, PID28416. Component ensemble probabilities and frozen policy bits exact; repeated full training/official replay claim 아님.
- [independent QA](../../artifacts/p1_core_tuning_20260906_v1/independent-qa.json), SHA `2a36caee9ee23135041fd96e1a1b4c4ac7c0677b692681057c9ed0e199340c6d`, PID38168. 실제 모델 설정/seed/hash/행/라벨/키/inner 선택/분모/CI/risk 재계산158검사.
- OOF SHA `9fc142914c9395ee7aa1a9139dea9316bcd2c91326845808931e735dea130c2c`; row-level OOF/models는 local-only이며 Git 대상 아님.
- [root 별도 산술·지문 QA](../parallel_core_training_20260906_v1/p1-root-independent-qa.json). 공식 점수와 현재 최고점 갱신은 이번 실험에서 측정하지 않았다.

아래는 학습 전에 고정한 계약 및 실행 내역이다. 학습 시작 2026-09-06 13:06:54 KST, launcher17636/worker22312; terminal 전 성능 조기 조회·변경0.

## Fixed question and budget

Can a bounded tree learning-rate/complexity/regularization change improve overall Q2/Q3/Q4 pooled F1 with the existing 80 features, weighting and union decoder? All periods have been exposed before: this is retrospective, not fresh confirmation. The new primary is 421,032 common outer rows; the previous Q3/Q4 pooled diagnostic remains a secondary column, not silently replaced in its original study.

Four complete policies: control O1/B3, O half-rate/double-rounds, O reduced complexity/stronger L2, or B reduced complexity/stronger L2. Each component threshold uses only the same earlier 60-day inner interval and existing grid/tie rule; the complete O OR B policy is selected on inner pooled F1. Only the winning altered component is then fit on outer training data. Unselected alternative policies have no outer result and will not be represented as evaluated.

The three B seeds mean three real models: 20260813, 20260829, 20260847. Fresh CPU2 baseline costs 24 fits, all alternative inner models 15, selected outer additions 0–9: **39–48 actual fits maximum**, with a 5,400-second worker/launcher cap. Estimate 35–50 minutes plus replay/QA. B-slow was excluded before training for budget. No stored CPU4 baseline OOF/model is reused, eliminating a training-resource comparator confound. Model `n_jobs=2`, environment limits and loaded native pools are enforced; GPU use is zero.

## Prior work and immutable scope

The prior [two-sided curve](../p1_tuning_twosided_20260906_v1/report-source.md) used a different split, single-seed B and same-rate 1,400-tree diagnostic. Its eight-settings-per-arm HPO proposal was not executed. It does not duplicate this chronological O1/B3 half-rate/coupled-regularization comparison. The prior curve did not prove that 700 trees is universally optimal.

Source functions and legitimate seeds/weights come from [tree lineage](../p1_champion_reconstruction_20260906_v1/tree-lineage.md). Train-only year-depth, unknown-year NaN/category behavior, 168-hour plateau cap, complete permitted-partition contextual scope, run-start ownership and 21-day purge are unchanged. This is not exact reproduction of the old 28.9-point preprocessing, a depth repair, a cell-policy change or a bracket experiment. Existing artifacts, locks, models and champion answers remain immutable.

The primary includes train-unsupported station-layer rows. Support counts and known/unseen slices, quarter declines, worst-day/cell effects and paired calendar-day bootstrap CI90/P(improvement) are reported as risks; no 0.8 probability hard gate or post-hoc row exclusion. Mean improvement can preserve a candidate without proving an official gain. No Public-score inversion, coefficients, old answer input, official test/sample, hidden labels, CSV, upload or automatic full fit.

Canonical artifacts will be `artifacts/p1_core_tuning_20260906_v1/`: terminal result, exact fresh-process replay, independent arithmetic QA. The sealed contract is `configs/experiments/p1_core_tuning_20260906_v1.json` and the new runner is `scripts/p1_core_tuning_20260906_v1/run.py`.

## Preflight and support

Runner synthetic tests: 15 PASS; dedicated independent-QA synthetic tests: 9 PASS; each focused Ruff check PASS. Synthetic tests perform no model fits. The initial Ruff undefined-selector finding was fixed before sealing/launch; no training artifact or sealed source was changed. Source SHA `4687b1e7273aeb22b686fd9acba3f7d19cd60416a6aa58f5832006cfe5a215bb`; [seal](seal.json) SHA `938063dc9c3c75aa98624e0bf9d5603b513da96cd3e2648f2ee0c93c76fd82f1`.

Source-derived outer support: Q2 has 26,062 unseen station-layer rows / 133,170; Q3 and Q4 have zero such rows. All remain in the denominator. This does not address deployment unseen-year depth metadata, which is a different limitation.

## Conditional handoff, not executed

Pre-result proposal: deployment could use a separately preregistered final train-only 60-day interval and the same nine-model inner selection, then four all-train models: 13 additional fits. Reusing the existing last Q4 inner selector would cost only four full fits but carries a past-window selection limitation. **After complete fixed-MS bridge, root decided not to execute either full branch in this cycle because composed pooled F1 did not improve.** The tree-only mean-improving variant, existing MS-TCN, champion b2f17 and fallback remain preserved. No coefficient or threshold was chosen from official scores.

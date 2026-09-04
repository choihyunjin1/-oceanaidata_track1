# P1 v27 FN event-archetype audit — design-only conclusion

The only simple axis with the same useful ordering direction in Q2, Q3, and Q4 is the conservative two-view score `min(probability_base, probability_peer)`. The proposed next candidate is therefore a **prefix-frozen ECDF rank consensus** over those two probabilities, followed by an inner-only add-only selector. This is not the v25 label-shift EM proposal: it performs no prevalence estimation or posterior-odds correction. No candidate code, historical fit, attempt lock, official/hidden read, CSV, or upload was produced.

## Scope and oracle boundary

- Historical OOF surface only: 421,032 rows across Q2/Q3/Q4.
- Anchor false negatives were defined as `label_base=1` and `current_router_prediction=0`.
- Truth-run phase, duration, and every precision/AUC below use oracle labels for **hypothesis discovery only**.
- No outer-fold statistic may set a future threshold. A future Q3 threshold must be selected only from an inner chronological tail of Q2; a future Q4 threshold only from an inner chronological tail of Q2+Q3.

## False-negative archetype

| fold | FN | onset | interior | end | singleton | duration >180m |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 1,575 | 9 | 1,562 | 3 | 1 | 1,574 |
| Q3 | 1,156 | 10 | 1,132 | 9 | 5 | 1,151 |
| Q4 | 568 | 6 | 559 | 2 | 1 | 567 |

The miss is overwhelmingly a long-event interior problem, not an onset/end problem. Yet causal prior-anchor density was weak/inconsistent, so this does not justify another morphology or run-extension rule.

Support concentration also changes materially: Q2 is dominated by S-ORS/I-ORS and L1/L2/L6; Q3 includes G-ORS and is L1/L2-heavy; Q4 is mostly I-ORS L1/L5. A fixed station/layer/quarter router is therefore rejected.

## Simple-axis direction audit among anchor-negative rows

| axis | Q2 AUC | Q3 AUC | Q4 AUC | decision |
|---|---:|---:|---:|---|
| `min(base, peer)` | 0.899851 | 0.888595 | 0.711688 | retain |
| base probability | 0.891288 | 0.883859 | 0.686342 | weaker single view |
| peer probability | 0.893492 | 0.890698 | 0.723972 | weaker robustness alone |
| e150 probability | 0.795624 | 0.487601 | 0.662649 | reject: Q3 collapse |
| peer/depth e150 alignment | 0.638062 | 0.408035 | 0.443506 | reject: reversal |
| absolute peer residual | 0.526029 | 0.478854 | 0.668003 | reject: only Q4 |
| causal 10m slope | 0.490960 | 0.477–0.508 range | 0.539622 | reject |
| causal 1h slope | 0.496805 | 0.477–0.508 range | 0.548995 | reject |
| causal 6h slope | 0.478081 | 0.477–0.508 range | 0.573145 | reject |

Raw-score calibration is not stable enough for a fixed cutoff. At each fold's top 0.1% by joint score, oracle precision was 68.22% in Q2, 68.60% in Q3, but only 18.52% in Q4. Prior-anchor conditioning improved Q3 but collapsed in Q4. Thus neither a pooled raw threshold nor a causal morphology conjunction is justified.

## v25 independence and duplicate audit

v25 recommended `prequential unlabeled label-shift EM odds correction` over the frozen probability vector. It explicitly targeted target-prevalence/calibration-state shift. The v27 proposal below is distinct: it uses only prefix-frozen empirical marginal ranks and conservative two-view intersection, with no EM, prevalence estimation, odds correction, station router, event morphology, or outer-label adaptation.

## Single proposed v27 candidate

**PREFIX_ECDF_BASE_PEER_MIN_CONSENSUS**

1. On each outer train prefix only, fit empirical CDFs for base and peer probabilities, pooled globally with a predeclared minimum-support station-layer fallback.
2. Freeze those CDFs before the outer fold. For every anchor-negative row compute `s=min(ECDF_base(p_base), ECDF_peer(p_peer))`.
3. On the last chronological 25% of the train prefix only, choose an add-only score cutoff that maximizes actual union F1 subject to positive inner delta, marginal precision above anchor F1/2, and changed-row share <=0.5%; ties choose the higher cutoff and fewer additions. If none qualifies, abstain.
4. Apply the frozen CDFs and cutoff unchanged to the outer fold. Preserve the anchor exactly; remove zero rows.
5. Require both Q3 and Q4 delta F1 nonnegative, dependent day-block CI stability, slice guards, and the governing prospective transport gate. Outer labels are evaluation-only.

Falsifier: abstention, negative Q3/Q4 delta, failure of either stability/slice guard, or failure of the transport gate closes this rank-consensus candidate. The audit does not claim it will pass; it is the sole nonduplicative candidate supported by a same-direction ordering signal.

## Access ledger

- historical OOF reads: 1
- model fits / candidate materializations / attempt locks: 0 / 0 / 0
- official / hidden / submission / upload reads or writes: 0 / 0 / 0 / 0

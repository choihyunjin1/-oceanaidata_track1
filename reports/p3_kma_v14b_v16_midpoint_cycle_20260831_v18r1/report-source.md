# P3 v18 midpoint recovery

## 결론

- Historical episode-blocked evaluation is a strict internal PASS, but the candidate is not deployable to the official rows and no submission CSV was produced.
- The evaluation-only recovery reproduced the frozen v18 candidate without model, weight, gate, or threshold changes: pooled delta RMSE `-0.004417861 m`, 4/6 improved bimonth blocks, episode CI90 upper `-0.001541916 m`, and block-by-station CI90 upper `-0.000853806 m`.
- Central expected score gain is `+0.070114717` points before the family transport penalty and `+0.020528663` points after the fixed `0.049586054` penalty.
- Independent QA passed 12/12 and evaluation accessed zero official rows.

## Materialization outcome

- The exactly-once materializer consumed its lock and then failed before constructing or writing a submission with `KeyError: anchor_time`.
- The frozen v16 harmonic component requires absolute anchor time. Official `test_index.csv` exposes only `case_id`, `station`, and `lead_h`; official context and the sealed test feature parquet also do not expose `anchor_time`.
- Replacing the harmonic time input, inferring it from row order, or changing the formula would change the frozen prediction contract. No such recovery was attempted.
- Delivery directory, CSV, hidden-truth access, and uploads remain zero.

## Evidence scope

The 182-case historical surface has been reused adaptively and is development evidence, not independent confirmation. The internal PASS is preserved as evidence of the formula on historical timestamps; the official deployment failure is preserved separately as a technical/deployability failure.

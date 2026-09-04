# P3 v36 terminal technical failure

## 결론

- `p3_delay_persistence_residual_cycle_20260901_v36` is **INVALID_TERMINAL_TECHNICAL_FAILURE** and must never be rerun under the same ID.
- The exactly-once lock was consumed, but failure occurred during feature construction before any model fit, outer prediction, metric, CI, or candidate ranking was produced. Fit count and exposed outer-score count are both `0`.
- The immutable science contract was not evaluated. This is not a scientific NO_GO.

## Cause

At historical surface position 21 (`anchor_id=12883`, `I-ORS`, block `07_08`), the fixed wind-speed path was constant. Its delay-coordinate point cloud therefore contained only exact-zero pairwise distances. SciPy's dense `minimum_spanning_tree` adapter treated those zero-weight entries as absent edges, returned zero rather than `n-1` MST edges, and triggered the sealed shape guard:

`ContractError: H0 persistence/MST lifetime contract differs`

The issue is limited to the numerical MST adapter. A mathematically exact deterministic Kruskal implementation can retain zero-weight edges without changing channels, delays, embedding dimension, lifetime definitions, model candidates, blend, folds, purge, or gates, but it requires a fresh exactly-once ID.

## Immutable evidence

- config SHA-256: `ad48a4ff053c663c0aefb90a8a563d57321ce358520bfd08d92693b73393556e`
- runner SHA-256: `3dc7ee1754eb153e1c8a64f31f500cbff56973a438a23a99ead8bc807f82e8de`
- consumed lock SHA-256: `4409f2dd6f73b34d6d70e385a0b47fca1f574e0a50104d08afb3b913849ca447`
- terminal `result.json`: absent
- evaluation arrays: absent
- artifact/report deletion or mutation: none
- official/hidden/test/sample/submission/CSV/upload access: `0`

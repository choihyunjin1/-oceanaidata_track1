# P1 v13 causal endpoint-visibility terminal report

`p1_v13_causal_endpoint_visibility_topology_addonly_20260901_v1` is a valid terminal `NO_GO_EXPLORATORY_ONLY` and exact no-op. The namespace must not be retried or retuned.

The fixed representation maps each station-layer's past 96 samples to upper/lower horizontal endpoint-visibility degree, span, span entropy, and degree persistence. Its mechanism is motivated by Luque et al., *Horizontal visibility graphs: exact results for random time series* (2009), https://arxiv.org/abs/0809.1906; this is a causal endpoint approximation, not a reproduction or source-derived performance claim. Repository audit found no prior P1 visibility-graph implementation. This differs from physical station/layer graph adjacency, v10 state-distance recurrence, MiniRocket/soft-symbolic operations, and run/duration decoders. No P2/P3 result or hyperparameter was transferred.

After v12's inner-to-outer failure, v13 prospectively added a hashed transport-stability veto before any execution: environments are fixed station x layer x chronological-half cells; supported cells require at least 10 proposals; at least two supported cells spanning both halves must each have precision strictly above `0.55`; any supported cell with TP `0` vetoes the threshold. This did not change the predeclared quantiles, 96-row window, model, seeds, or pooled Wilson gate.

- Focused pytest `5 passed`; Ruff PASS.
- Two real zero-operation preflights were byte-identical, SHA-256 `8933cfa1967260745e002a500bd5a0afab2872689b0f6ce356609022c578b28a`; artifact/lock stayed absent.
- Feature nonzero share `0.8530035817928534`; all ten variances passed the frozen support gate; pre-execution QA PASS.
- Exactly 9 distinct fits, runtime `91.672 s`, outer target reads before all seals `0`.
- No threshold passed the transport veto in Q2, Q3, or Q4; all sealed `chosen=null`. Additions/removals `0/0`.
- Pooled incumbent=candidate F1 `0.8604836038423319`, TP/FP/FN `12989/1146/3066`; all fold deltas `0`.
- CI90 `[0,0]`; nominal/transport points `0/0`; long-event interior recall `0.8107135718568859`, offset `0.6477211796246649`, drift `0.6595061728395062`, all unchanged.
- Official/test/sample/submission/hidden reads, CSV, upload: `0`.

Lifecycle-independent QA recomputed counts/F1, long-event recall, add-only/removal-zero, seal hashes, nine unique model hashes, outer isolation, schema, and access-zero counters: `PASS`. Config/runner/completion/lock hashes are `c724d938c19fcbd8018cbce7433b48af31220ca45bb96520ec21acffcedd2e33`, `84d0814f2a8a353bf40b149912e1bd796f904f0150a2a25139dd58176a6be527`, `da3771f3d512f531a0746637dc4a0fa8240d02250e5ccc4bf176363da0c58712`, `2d1cfebb66858f94e8fab4105b26d7e886eba45d7debfbcb6fb46a4f5287a4f3`. Result SHA-256: `187cc63250a1b541b15efc52548d59b1a652816f4cf8bd289c0bfc79e022e871`.

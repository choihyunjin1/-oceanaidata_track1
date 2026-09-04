# P1 v55 Hawkes event-hazard research-frontier audit

## Terminal decision

`NO_GO_ZERO_FIT_RESEARCH_FRONTIER_EVENT_HAZARD_NOT_IDENTIFIABLE`.

The one audited mechanism was a causal station-layer exponential Hawkes intensity over anomaly-event onsets. Hawkes's original paper establishes self- and mutually-exciting point-process dynamics; it does not establish anomaly clustering, deployment observability, or P1 benefit ([Hawkes, Biometrika 1971](https://academic.oup.com/biomet/article-abstract/58/1/83/224809)).

An exact Hawkes implementation is absent from the repository, but the scientifically meaningful event stream is not available at deployment: past ground-truth anomaly onsets are themselves target information. Replacing them with frozen anchor onsets makes the score a deterministic memory-kernel transform of information already used through `since_anchor`, signal run length, rolling/lagged probabilities, semi-Markov duration/state decoders, recurrence/laminar summaries, and return-time/event proposals. That is a kernel variant, not a new observed scientific state.

The semantic and support gates therefore fail before executable preflight. No threshold, decay half-life, candidate, seed, split, or action budget was selected, and no two-preflight or historical execution was authorized. This preserves v53's no-fresh-window finding and v54's no-overlooked-candidate finding without forcing another exposed-surface variant.

Minimum new evidence is a competition-allowed deployment-time event or maintenance stream independent of the anomaly label, with authenticated timestamps and train/deployment coverage; alternatively, a fresh labeled chronological holdout paired with a genuinely non-reconstructible observable.

Operations: train rows 0, target reads 0, fits/optimizer/actions/removals 0, official/test/sample/submission/hidden/CSV/upload 0.

# v25 oracle audit technical recovery

The first `result.json` is preserved as `INVALID_EVENT_GEOMETRY_DATETIME_UNIT`. Pandas supplied integer timestamps in microseconds while the initial event-cadence comparison used a nanosecond constant. This made each positive row appear to be a singleton. Candidate count, model fits, threshold searches, attempt locks, official reads, hidden reads, CSVs, and uploads were all zero.

The corrected run changes only event geometry cadence detection to `time.diff().total_seconds() == 600`. It writes a separate `result.corrected.json`; the initial result is not overwritten. Station/layer/quarter, missingness, probability-bin, and residual-bin calculations did not depend on this cadence comparison.

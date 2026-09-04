# P1 v49 causal Thorpe-overturn state audit

Audience: P1 experiment review. Date: 2026-09-01 KST. This bounded audit used only the authorized train prefix and the columns `station, layer, time, temp, psal, depth`; labels and outer targets were not read.

## Decision

`NO_GO_ZERO_FIT_SINGLE_STATION_PROFILE_SUPPORT_AND_SEMANTIC_RECONSTRUCTIBILITY`. No runner, preflight, lock, fit, or action was created.

Thorpe displacement has a genuine physical mechanism: a measured vertical profile is reordered to a stable monotonic profile, and RMS displacement summarizes overturn scale ([Lorke and Wuest, JGR Oceans 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JC001154)). The source concerns resolved stratified profiles and does not establish P1 anomaly precision or transport.

The P1 support geometry fails the prospective requirement. Among 294,278 prefix rows, 291,914 have finite temperature, salinity, and depth. However, simultaneous profiles with at least three finite levels occur at only S-ORS: 53,319 such profiles. G-ORS has at most one level and I-ORS at most two. A vertical reorder observable would therefore be unavailable at two of three stations and would reinforce the existing S-only action collapse rather than distinguish cross-station opportunity.

It is also semantically reconstructible. P1 already contains a label-free stratification peer gate based on simultaneous layer spread/gradient/coherence, an executed exact vertical bracket/rank family, and v21's explicit finding that EOS/density transformation repeats the existing temperature-salinity-depth and TEOS/profile family. Temperature-only Thorpe inversions are not an escape hatch because salinity can compensate temperature inversions; using density returns to the closed EOS family.

The candidate was fixed before rejection at four overturn summaries, at least three finite depths, at least 10,000 profiles in each of at least two stations, three seeds/three fits, unchanged thresholds, v28/v33, and add-only removal=0. The support gate failed before readiness, so two READY preflights and exactly-once execution were inapplicable.

Counters: fits=0, optimizer steps=0, preflights=0, target reads=0, actions=0, removals=0. Official, hidden, test, sample-submission, submission, CSV, and upload accesses were all zero.

The remaining gap is structural: the original schema contains no cross-station vertical scientific-state observable because G-ORS is single-layer and I-ORS has at most two simultaneous finite levels. Resolving it requires an independently observed cross-station driver or organizer-authorized external physical covariate, neither available inside this train-only cycle.

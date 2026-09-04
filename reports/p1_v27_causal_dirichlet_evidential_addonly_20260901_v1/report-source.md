# P1 v27 causal Dirichlet evidential add-only cycle

## Terminal decision

`NO_GO_EXPLORATORY_ONLY`, immutable and no retune. The distinct objective maps a small causal network to nonnegative two-class evidence, treats evidence plus one as Dirichlet concentration, and minimizes expected squared error plus predictive variance and a fixed wrong-class-to-uniform KL term. The action score is the expected positive-class probability; Dirichlet uncertainty is diagnostic only and never selects a threshold.

[Sensoy et al. (NeurIPS 2018)](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) motivate learning a Dirichlet distribution over class probabilities from evidence. This is mechanism support only, not a P1 performance claim.

Repository fingerprinting found no executed P1 evidential, Dirichlet, or subjective-logic classifier. v27 has no prototype/hypersphere, domain adversary, pairwise ranking, cross-view Barlow objective, score recalibration, or outer uncertainty selection. Its exact objective is now closed.

## Protocol and prospective guard

- two real train-only zero-operation preflights were byte-identical: 2,968 bytes, SHA-256 `d9b911cc3135bc20fd04acfbf9570aa11604130afd21a3b8ea31fcf5e96d58e9`; fits/targets/official/CSV/uploads were zero.
- fixed causal temperature state features: current value, lags 1/6/36, first difference, absolute difference, acceleration, and elapsed-gap support; station-layer reset and ns cutoffs are inherited from the authenticated scaffold.
- fixed 8-unit tanh evidential network, 12 epochs, KL coefficient `0.01`, 3 seeds x Q2/Q3/Q4 = exactly 9 fits; no sweep, retry, outer tuning, or outer row in training.
- the prospective guard amendment SHA-256 `6d0f6f21aaa72410ad84d6a42e400b94a3e681f65670f81a02d464989b592383` requires both chronological halves, at least two station-layer identities and two stations, TP>0 and precision>0.55 in every supported environment, plus the unchanged pooled Wilson-90 LCB and minimum-addition gates.

Q2 selected inner quantile `0.9975`: 141 proposals, precision `0.900709`, Wilson LCB `0.851444`, with both halves, two station-layer identities, and two stations. On isolated outer Q2 it produced 332 additions but only 4 TP (precision `0.012048`), concentrated in G-ORS/L1 and I-ORS/L1. Q3 and Q4 failed the amended inner guard and remained exact no-ops. This is strong additional evidence that even the identity-diversified inner veto does not ensure station-layer-quarter transport.

## Canonical metrics

| metric | value |
|---|---:|
| incumbent / candidate pooled F1 | `0.8604836038423319 / 0.8513858855907215` |
| candidate TP / FP / FN | `12993 / 1474 / 3062` |
| additions / addition TP / precision | `332 / 4 / 0.012048192771084338` |
| anchor removals | `0` |
| raw F1 delta / block CI90 | `-0.009097718251610432 / [-0.01256378494600947, -0.006524518695906073]` |
| nominal / transport-adjusted points | `-0.24182091886045298 / -0.07254627565813589` |
| long-event interior recall anchor / candidate | `0.8107135718568859 / 0.8107801985475381` |
| long-event boundary recall anchor / candidate | `0.779835390946502 / 0.7818930041152263` |
| offset / drift recall | `0.6479892761394102 / 0.659753086419753` |
| fits / runtime seconds | `9 / 24.375` |

Fold F1 deltas were Q2 `-0.023923701473487435`, Q3 `0.0`, and Q4 `0.0`. The worst pooled action slice was G-ORS/L1: 159 additions and F1 delta `-0.0649881888072843`. Official/test/sample/submission/hidden reads, CSVs, uploads, outer-target reads before every seal, outer rows in training, and anchor removals were zero.

## QA and hashes

- focused pytest `4/4 PASS`; Ruff `PASS`; post-terminal lifecycle/hash QA all checks `PASS`
- config `a281aafdd4d479ea492c9b9bd4c929394d7a6a947c59eb3f7154275a3ac8a554`
- runner `02d82470f60bc8f856f9218cbe80c5042b7b9f2b41165f7ef246088684fb7143`
- completion `059a469d793e9d92c6ceda00a82bdcd5e32e601f55241fa73b39dfc36deaf96c`
- lock `fa914f4711485820a22a918138c003618281444af0c16dc12bbdb443381f609e`
- result `df57fba75d34a03b1b2956d0c145a4efef554d3faccabff087a989d3aa0e15b7`

The evidential objective, feature set, architecture, KL coefficient, quantiles, budget, and guard application are closed and must not be retuned.

# P1 robust-subspace block-conformal cycle — terminal report

## Conclusion

The label-free feasibility audit passed, but the separately sealed historical falsification was a strict no-op and therefore `NO_GO_RESEARCH_ONLY`. The frozen e-BH rule rejected no validation blocks, added zero rows, removed zero champion rows, and produced exactly the champion metrics. Retuning q, the e-value transform, block length, or run rule is prohibited.

## Label-free audit

`p1_robust_subspace_block_conformal_feasibility_20260901_v1` completed with three input-only robust subspace fits, zero supervised fits, zero target reads, and zero official-interface reads. For Q2/Q3/Q4 and block lengths 72/144/288, rejected blocks, empirical false-alarm rate, proposal share, and cell concentration were all zero. Every sensitivity decision agreed, so the separate performance namespace was authorized.

## Historical falsification

Predictions for Q2, Q3, and Q4 were sealed before historical targets were opened. Candidate predictions were bitwise identical to the champion because additions were zero.

| Surface | Rows | F1 | TP | FP | FN | Additions | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 | 133,170 | 0.778413575375 | 3,945 | 821 | 1,425 | 0 | 0 |
| Q3 | 176,738 | 0.897058823529 | 5,307 | 128 | 1,090 | 0 | 0 |
| Q4 | 111,124 | 0.909024568232 | 3,737 | 197 | 551 | 0 | 0 |
| Pooled | 421,032 | 0.860483603842 | 12,989 | 1,146 | 3,066 | 0 | 0 |

The 2,000-replicate paired block bootstrap CI90 was `[0, 0]`; every station-layer slice delta was zero. Nominal and transport-adjusted planning points were both zero. Runtime was 15.953 seconds with three input-only fits, zero supervised fits, and zero anchor removals.

## Integrity and access

- feasibility result SHA-256: `09843761d5a7e15fd82984735938966f2e65ea5949d5e4f92073d1a46d6aceb0`
- falsification result SHA-256: `e5e13b1f8a99188f3cdecfd9955c5e5a8e428543adc52adc3576e25ada54efb3`
- prediction-completion SHA-256: `cba1ba451a1506bfb0cc9dd108ecb4a69a82645eacf3a96f67c8b1ed9daa57c5`
- config / runner / lock: `a65f97b16a5a7ce441fee3bb5ae907f5ea24d8e24defec454e399913ac3604c9` / `f3420d03171a18b336915d74a3623d0edd95d4afb4763d6f6afcd4929ffc6019` / `73ede0b2c9607f2289715ce54718117bccdee0b842bb8a29260bc43eaa82aaea`
- official test, sample submission, hidden data, candidate CSV, and upload operations: zero.

Post-terminal focused pytest was 6/6 PASS and Ruff was PASS.

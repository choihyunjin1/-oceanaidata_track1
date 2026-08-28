# Claim-to-source ledger

| Claim | Evidence | Confidence / caveat |
|---|---|---|
| P2 alpha40 scored Public RMSE 0.445147 and 27.747847 points | `official_score_receipt.json`; authenticated OCN-02 result card | High for the public score; Private metric is not yet visible |
| P2 file is the preregistered 26,061-row alpha40 candidate | `p2_oas40_readiness.json`; candidate SHA-256 `6e28...b96` | High; exact hash and row/schema QA were independently repeated |
| P1 verifier failure is statistical support scarcity, not compute scarcity | `p1_frozen_direct_event_verifier_blocked_20260828_v2/aggregate_metrics.json` | High for this frozen generator-verifier pairing; does not rule out every P1 architecture |
| P1 NCAD-inspired synthetic TCN does not transport safely | `p1_ncad_synthetic_long_event_20260828_v1/result.json` | High on the exposed local surfaces; no official submission inference |
| P3 past-only compact linear and TSMixer candidates fail at long leads | NLinear result plus prior TSMixer result | High on the fixed 181-case historical surface; does not prove an absolute model ceiling |
| P3 ERA5 route has not scientifically failed | `p3_era5_context_transfer_terminal_20260828.json` | High: download/preflight passed, but zero model fits occurred because CatBoost import failed |
| Contextual outlier exposure is a defensible P1 direction | IJCAI 2022 NCAD paper and official proceedings page | Literature support only; local implementation is inspired by, not an exact reproduction |
| P3 next high-value direction is future/exogenous forcing | TimeXer, TiDE, and DLinear primary/official sources plus local long-lead failure | Mechanistic inference; requires a fresh preregistered experiment |

## External sources

1. IJCAI 2022, Neural Contextual Anomaly Detection for Time Series: https://www.ijcai.org/proceedings/2022/394
2. DLinear official implementation: https://github.com/honeywell21/DLinear
3. TiDE primary paper: https://openreview.net/pdf?id=pCbC3aQB5W
4. TimeXer NeurIPS 2024 paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/0113ef4642264adc2e6924a3cbbdf532-Paper-Conference.pdf

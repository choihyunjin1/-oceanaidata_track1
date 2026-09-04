# P3 v10 claim-source ledger

| claim | evidence | status |
|---|---|---|
| Train-only downweighting is preferable to deleting validation outliers. | Runner receipts record q90 residual cutoffs, positive weights, and zero validation-row deletion. | Directly measured |
| All three direct robust-base retrains fail the sealed public-transport gate. | `result.json`: pooled deltas +0.043353, +0.003154, +0.016880 m; 0/3 PASS. | Directly measured |
| Public reversal risk must be subtracted from internal expected points. | `reports/public_transport_calibration_20260831_v1/calibration.json`: penalty 0.321905690 points. | Authoritative local calibration |
| Robust loss alone does not repair the observed transport reversal. | All v10 candidates regress before the public penalty is applied. | Supported for tested families only |

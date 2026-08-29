# p2_nested_pls_capacity_grid_20260829_v1

- Decision: `NO_GO_CLOSE_FAMILY`
- Pooled delta RMSE: `-0.002041992` C
- Day-bootstrap upper q95: `-0.001051630` C
- PLS fits: `84`; grid points: `243 x 3 outer folds`
- Runtime: `1902.745` seconds
- Official hidden/test/sample/submission values read: `0 rows`; CSV/upload: `false`.

## Outer selections

- `2024_sep_oct`: `r1_ridge1e-03_q0.975_cap0.050_s0.75`; inner delta `0.000000000`; eligible `False`
- `2025_jul_aug`: `r1_ridge1e-04_q0.990_cap0.075_s1.00`; inner delta `-0.000603241`; eligible `True`
- `2025_nov_dec`: `r2_ridge1e-04_q0.975_cap0.050_s0.75`; inner delta `-0.000035568`; eligible `True`

## Gate checks

- `all_inner_selections_eligible`: `False`
- `pooled_delta`: `True`
- `2024_sep_oct`: `True`
- `two_of_three_folds`: `False`
- `worst_fold`: `False`
- `all_layers`: `True`
- `day_bootstrap_upper`: `True`
- `oas_cosine`: `True`
- `historical_cosine`: `True`
- `correction_p99`: `True`
- `correction_rms`: `True`

The three outer blocks are sealed against their own labels but historically exposed; this is local proxy evidence only.

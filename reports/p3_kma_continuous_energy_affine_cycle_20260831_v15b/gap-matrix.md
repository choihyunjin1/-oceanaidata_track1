# Gap matrix — P3 continuous-energy affine v15b

| Requirement | Observed | Decision |
|---|---:|---|
| Pooled RMSE improvement | -0.001776322 m | PASS |
| Improved blocks >=4/6 | 3/6 | FAIL |
| Episode CI90 upper <0 | +0.003412292 m | FAIL |
| Block×station CI90 upper <0 | +0.004214669 m | FAIL |
| Worst station×lead <=+0.01m | +0.012591432 m | FAIL |
| Raw LCB >=0.059586054 points | 0.000000000 | FAIL |
| Calibrated LCB >=0.01 points | -0.049586054 | FAIL |

The learned energy slope expanded in later prefixes and increased uncertainty. The candidate remains a scientific NO_GO and did not open official inputs.

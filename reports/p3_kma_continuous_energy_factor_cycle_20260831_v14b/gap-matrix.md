# Gap matrix — P3 continuous-energy KMA factor v14b

| Requirement | Observed | Gap | Decision |
|---|---:|---:|---|
| Pooled RMSE improvement | -0.003547692 m | none | PASS |
| Improved bimonth blocks | 4/6 | none | PASS |
| Episode CI90 upper < 0 | -0.000322053 m | none | PASS |
| Block×station CI90 upper < 0 | +0.000365792 m | 0.000365792 m | FAIL |
| Worst station×lead <= +0.01 m | +0.007404054 m | none | PASS |
| Changed share <= 1/3 | 0.333333 | none | PASS |
| Raw LCB >= 0.059586054 points | 0.000000000 | 0.059586054 | FAIL |
| Calibrated LCB >= 0.01 points | -0.049586054 | 0.059586054 | FAIL |

The central expected improvement was +0.056304493 points, just below the family raw threshold, but the governing value is the conservative bootstrap LCB. The continuous-energy KMA direction is therefore closed without official materialization.

# Gap matrix — P3 annual-harmonic shrink v16

| Requirement | Observed | Decision |
|---|---:|---|
| Pooled RMSE improvement | -0.005101662 m | PASS |
| Improved blocks >=4/6 | 4/6 | PASS |
| Episode CI90 upper <0 | -0.002450083 m | PASS |
| Block×station CI90 upper <0 | -0.001724512 m | PASS |
| Worst station×lead <=+0.01m | +0.002002339 m | PASS |
| Changed share <=1/3 | 0.267399 | PASS |
| Raw LCB >=0.059586054 points | 0.027369283 | FAIL by 0.032216771 |
| Calibrated LCB >=0.01 points | -0.022216771 | FAIL by 0.032216771 |

Central expected improvement was +0.080967148 points and central-minus-family-penalty was +0.031381094 points. The governing gate uses the conservative local LCB, so no official materialization occurred.

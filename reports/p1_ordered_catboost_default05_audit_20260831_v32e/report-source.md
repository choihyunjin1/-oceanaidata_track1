# P1 ordered CatBoost default-0.5 audit

Conclusion: **TERMINAL NO_GO**. The immutable v32a probabilities were converted once with CatBoost's fixed default threshold 0.5. The action mask and code/config hashes were sealed before historical truth attachment. No model was refit and no threshold was searched.

- Q2 versus tabular: delta F1 +0.025982.
- Q3 versus tabular: delta F1 -0.099013.
- Q4 versus tabular: delta F1 -0.067660.
- Pooled versus tabular: delta F1 -0.048750; CI90 [-0.075736, -0.023130], P(improve)=0.0006.
- Q3/Q4 versus E150: delta F1 -0.090549; CI90 [-0.129982, -0.055708], P(improve)=0.
- Changed 2,887 rows: 434 additions and 2,453 anchor removals; maximum station-layer-fold concentration 9.3523%.
- Linear score translation: center 27.613650, CI90 [26.896420, 28.294582]. This is an empirical translation, not a guarantee.

All strict gates failed. Result-based threshold retry is prohibited. Official, hidden, sample, submission, CSV, and upload counts are zero.

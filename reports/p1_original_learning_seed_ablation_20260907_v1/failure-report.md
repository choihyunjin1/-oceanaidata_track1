# Technical failure, not a scientific result

The first Q3 O_slow model finished `_fit_model`, then the validation code called `get_booster()` on `DeterministicTabularClassifier` instead of its `.model` backend. AttributeError occurred before serialization and fit-receipt append. Terminal receipt says persisted fits=0, but actual completed training=1. No performance result or candidate answer was produced. Runtime76.031s. Original source/config/lock/terminal/log preserved.

New v2 fixes only backend inspection and adds synthetic verification of that exact accessor. Scientific parameters unchanged; 9 new fits counted separately, so expected cumulative production work is10 including the lost v1 fit. No automatic restart of v1, no active P3 change.

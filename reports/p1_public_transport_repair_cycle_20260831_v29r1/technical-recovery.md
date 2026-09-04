# P1 v29r1 metric-only recovery contract

v29 completed both sealed prediction fits but failed before metric evaluation because the otherwise unchanged `bootstrap_probability_improved_minimum_inclusive=0.8` field was omitted from its config. v29r1 adds only that field, reuses the v29 sealed NPZ at fixed SHA-256, performs zero additional fits and zero prediction writes, and evaluates exactly once under a new lock. The v29 lock and artifacts remain unchanged.

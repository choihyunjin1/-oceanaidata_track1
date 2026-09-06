"""Independent paired station-episode RMSE bootstrap from grouped residual sums."""

import math

import numpy as np


def independent_bootstrap(frame, settings):
    y, b, c = (frame[name].to_numpy(float) for name in ("target_hs", "control", "candidate"))
    if not len(y) or not np.isfinite(np.column_stack((y, b, c))).all():
        raise ValueError("finite aligned nonempty predictions required")
    groups = {}
    for i, key in enumerate(zip(frame.station, frame.episode_id, strict=True)):
        groups.setdefault(key, []).append(i)
    if len(groups) < 2:
        raise ValueError("at least two station-episode clusters required")
    statistics = []
    for indices in groups.values():
        statistics.append(
            (
                math.fsum(float(b[i] - y[i]) ** 2 for i in indices),
                math.fsum(float(c[i] - y[i]) ** 2 for i in indices),
                len(indices),
            )
        )
    statistics = np.asarray(statistics)
    rng = np.random.default_rng(settings["seed"])
    deltas = []
    for _ in range(settings["resamples"]):
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        sse_b, sse_c, rows = np.dot(weights, statistics)
        deltas.append(math.sqrt(sse_c / rows) - math.sqrt(sse_b / rows))
    return {
        "n_rows": len(y),
        "n_clusters": len(groups),
        "ci90": np.quantile(deltas, settings["ci_quantiles"]).tolist(),
        "p_improve": float(np.mean(np.asarray(deltas) < 0)),
        "resamples": len(deltas),
        "seed": settings["seed"],
    }

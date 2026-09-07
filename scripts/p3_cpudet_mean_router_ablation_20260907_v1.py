"""Aggregate-only, zero-fit ablation of the frozen CPU historical router."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from p3_numeric_cpudet_noshrink_20260907_v1_internal import bootstrap, compare, save, sha
from threadpoolctl import threadpool_limits


def mean_prediction(single, multi, persistence, leads):
    """Same original shrink; remove only learned component weights."""
    a, b, p, lead = map(np.asarray, (single, multi, persistence, leads))
    if not (a.shape == b.shape == p.shape == lead.shape):
        raise ValueError("component shapes differ")
    if not all(np.isfinite(x).all() for x in (a, b, p)):
        raise ValueError("nonfinite component")
    out = (a + b) * 0.5
    active = np.isin(lead, [12, 18, 24])
    out[active] = out[active] * 0.8 + p[active] * 0.2
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evaluate", "qa"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    cfg = json.loads(args.config.read_text())
    assert sha(cfg["source"]) == cfg["source_sha256"]
    f = pd.read_parquet(cfg["source"])
    assert len(f) == cfg["expected_rows"]
    assert not f.duplicated(["fold", "anchor_id", "station", "lead_h"]).any()
    assert f.groupby(["fold", "anchor_id"]).size().eq(6).all()
    assert f.episode_id.notna().all()
    f["noshrink"] = mean_prediction(f.single_prediction, f.multi_prediction, f.persistence, f.lead_h)
    # Historical helper column is named noshrink; here it holds the router ablation WITH shrink.
    short = f.lead_h.isin([3, 6, 9])
    assert np.allclose(f.loc[short, "noshrink"], f.loc[short, "final_prediction"], atol=1e-14, rtol=0)
    metrics = {"pooled": compare(f), "bootstrap": bootstrap(f, cfg["bootstrap"]),
               "low_wave_hs0_lt_1_7": compare(f.loc[f.current_hs < 1.7]),
               "tables": {key: {str(v): compare(p) for v, p in f.groupby(key)}
                          for key in ["fold", "station", "lead_h"]}}
    if args.mode == "qa":
        old = json.loads((args.out / "result.json").read_text())
        assert old["pid"] != os.getpid() and old["config_sha256"] == sha(args.config)
        assert old["metrics"] == metrics
        # Independently calculate each row using scalar arithmetic and math.fsum SSE.
        import math
        candidate_sse = math.fsum(
            (target - ((a + b) / 2 if lead < 12 else .8 * ((a + b) / 2) + .2 * p)) ** 2
            for target, a, b, p, lead in zip(f.target_hs, f.single_prediction,
                                            f.multi_prediction, f.persistence, f.lead_h, strict=True)
        )
        assert math.isclose(candidate_sse, metrics["pooled"]["candidate_sse"], rel_tol=1e-13)
        save(args.out / "independent-qa.json", {"status": "PASS", "pid": os.getpid(),
             "result_sha256": sha(args.out / "result.json"), "scalar_candidate_sse": candidate_sse,
             "new_fits": 0, "official_rows": 0, "seconds": time.monotonic() - start})
    else:
        save(args.out / "result.json", {"status": "RETROSPECTIVE_INTERNAL_COMPLETE",
             "pid": os.getpid(), "config_sha256": sha(args.config),
             "source_sha256": sha(cfg["source"]), "metrics": metrics, "new_fits": 0,
             "official_rows": 0, "hidden_rows": 0, "seconds": time.monotonic() - start})
    print(json.dumps({"mode": args.mode, "pooled": metrics["pooled"], "bootstrap": metrics["bootstrap"]}))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()

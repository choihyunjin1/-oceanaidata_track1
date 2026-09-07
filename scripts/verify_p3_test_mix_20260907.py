"""Read-only two-thread OOF test-mix diagnostic; never trains or writes answers."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa
from threadpoolctl import threadpool_limits

pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(path, test_shares):
    columns = ["station", "episode_id", "anchor_id", "lead_h", "current_hs",
               "target_hs", "persistence", "final_prediction"]
    frame = pd.read_parquet(path, columns=columns)
    assert len(frame) == 103602
    assert not frame.duplicated(["station", "anchor_id", "lead_h"]).any()
    assert frame.groupby(["station", "anchor_id"]).size().eq(6).all()
    assert frame.groupby(["station", "anchor_id"]).current_hs.nunique().eq(1).all()
    assert not frame.isna().any().any()
    truth, base, persistence, hs0 = [frame[k].to_numpy(float) for k in
                                   ("target_hs", "final_prediction", "persistence", "current_hs")]
    assert np.isfinite(np.column_stack([truth, base, persistence, hs0])).all()
    candidate = base.copy()
    long_leads = frame.lead_h.isin([12, 18, 24]).to_numpy()
    candidate[long_leads] = (base[long_leads] - 0.2 * persistence[long_leads]) / 0.8
    base_sq, candidate_sq = (base-truth)**2, (candidate-truth)**2
    bins = np.searchsorted([1.6, 1.7, 1.9, 2.4], hs0, side="right")
    shares = np.bincount(bins, minlength=5) / len(frame)
    assert np.all(shares > 0)
    weight = (test_shares / shares)[bins]
    grouped = frame[["station", "episode_id"]].copy()
    grouped["weight"] = weight
    grouped["base_sse"] = weight * base_sq
    grouped["candidate_sse"] = weight * candidate_sq
    cluster = grouped.groupby(["station", "episode_id"], sort=True).sum().to_numpy()
    rng = np.random.default_rng(20260907)
    delta = []
    for _ in range(2000):
        sample = cluster[rng.integers(0, len(cluster), len(cluster))].sum(axis=0)
        delta.append(float(np.sqrt(sample[2]/sample[0])-np.sqrt(sample[1]/sample[0])))
    # Independent grouped-bin aggregation must equal the direct row-weighted formula.
    alt = np.array([[base_sq[bins == i].mean(), candidate_sq[bins == i].mean()]
                    for i in range(5)])
    weighted = np.sqrt(np.array([weight @ base_sq, weight @ candidate_sq])/weight.sum())
    assert np.allclose(weighted, np.sqrt(test_shares @ alt), rtol=0, atol=1e-12)
    table = [{"bin": i, "oof_rows": int((bins == i).sum()), "oof_share": float(shares[i]),
              "test_share": float(test_shares[i]), "weight": float(test_shares[i]/shares[i]),
              "base_rmse": float(np.sqrt(alt[i, 0])), "candidate_rmse": float(np.sqrt(alt[i, 1])),
              "delta": float(np.sqrt(alt[i, 1])-np.sqrt(alt[i, 0]))} for i in range(5)]
    return {"path": str(path), "sha256": sha(path), "rows": len(frame),
            "anchors": int(frame.groupby(["station", "anchor_id"]).ngroups),
            "clusters": len(cluster), "oof_below_1_5_rows": int((hs0 < 1.5).sum()),
            "base_sse": float(base_sq.sum()), "candidate_sse": float(candidate_sq.sum()),
            "weighted_base_rmse": float(weighted[0]), "weighted_candidate_rmse": float(weighted[1]),
            "weighted_delta": float(weighted[1]-weighted[0]), "bins": table,
            "bootstrap": {"resamples": 2000, "seed": 20260907, "sort": True,
                          "method": "uniform station-episode resampling; fixed full-sample bin weights",
                          "ci90": np.quantile(delta, [.05, .95]).tolist(),
                          "p_improve": float(np.mean(np.asarray(delta) < 0))},
            "short_leads_exact_unchanged": bool(np.array_equal(base[~long_leads], candidate[~long_leads])),
            "independent_bin_vs_row_formula_pass": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    with threadpool_limits(limits=2):
        context = pd.read_parquet(args.context, columns=["case_id", "step_minute", "hs"],
                                  filters=[("step_minute", "==", 0)])
        assert len(context) == 200 and context.case_id.nunique() == 200
        hs = context.hs.to_numpy(float)
        assert np.isfinite(hs).all() and (hs >= 1.5).all()
        bins = np.searchsorted([1.6, 1.7, 1.9, 2.4], hs, side="right")
        counts = np.bincount(bins, minlength=5)
        cpu, gpu = analyze(args.cpu, counts/200), analyze(args.gpu, counts/200)
    assert abs(cpu["base_sse"]-48091.9523451907) < 1e-7
    assert abs(cpu["candidate_sse"]-47546.9931632454) < 1e-7
    expected = {"weighted_base_rmse": .641526, "weighted_candidate_rmse": .646559,
                "weighted_delta": .005033}
    comparisons = {k: {"observed": cpu[k], "fable_rounded": v,
                       "match_at_6_decimals": abs(cpu[k]-v) <= 5.1e-7} for k, v in expected.items()}
    result = {"status": "RECALCULATED", "new_fits": 0, "hidden_rows": 0, "uploads": 0,
              "thread_limit": 2, "pid": os.getpid(), "script_sha256": sha(Path(__file__)),
              "context": {"sha256": sha(args.context), "selected_case_rows": len(context),
                          "columns_read": ["case_id", "step_minute", "hs"],
                          "bin_counts": counts.tolist(), "shares": (counts/200).tolist()},
              "declared_edges": [1.5, 1.6, 1.7, 1.9, 2.4, "infinity"],
              "bin_note": "searchsorted matches Fable digitize; any OOF below 1.5 is counted explicitly",
              "cpu": cpu, "gpu": gpu, "fable_cpu_point_comparison": comparisons,
              "fable_cpu_ci90": [.002354, .007736], "fable_cpu_p_improve": .001,
              "caveats": ["Retrospective descriptive covariate-shift diagnostic, not official outcome.",
                          "Assumes conditional squared-error transport within hs0 bins.",
                          "No onset/station/season joint reweighting; hidden Public membership not accessed.",
                          "Weights and 200-case test mix held fixed in bootstrap; does not include their uncertainty."],
              "seconds": time.monotonic()-started}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"cpu": {k: cpu[k] for k in ["weighted_base_rmse", "weighted_candidate_rmse", "weighted_delta", "bootstrap", "oof_below_1_5_rows"]},
                      "gpu": {k: gpu[k] for k in ["weighted_delta", "bootstrap", "oof_below_1_5_rows"]},
                      "context_bins": counts.tolist(), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()

"""Frozen zero-fit smoothing plus projection of freshly generated L120 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from p2_final_day_materialize_20260907_v1 import BASE, endpoints, sha
from p2_final_day_projection_20260907_v1 import project_profiles, project_profiles_vectorized

NAME = "submission_p2_L120_s3_smooth7_proj.csv"


def smooth7(frame, prediction):
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("finite aligned predictions required")
    keyed = frame[["station", "layer", "time"]].copy()
    keyed["time"] = pd.to_datetime(keyed.time, utc=True)
    if keyed.duplicated().any():
        raise ValueError("duplicate keys")
    keyed["position"] = np.arange(len(keyed))
    out = values.copy()
    for _, group in keyed.groupby(["station", "layer"], sort=False):
        group = group.sort_values("time")
        times = pd.DatetimeIndex(group.time)
        grid = pd.date_range(times.min(), times.max(), freq="10min")
        if not times.isin(grid).all():
            raise ValueError("timestamps not on shared ten-minute grid")
        series = pd.Series(values[group.position.to_numpy()], index=times)
        smoothed = series.reindex(grid).rolling(7, center=True, min_periods=1).mean()
        out[group.position.to_numpy()] = smoothed.reindex(times).to_numpy()
    assert np.isfinite(out).all()
    return out


def base_provenance(package, source):
    """Require this run's successful QA, not equality with a historic prediction."""
    actual = sha(source)
    qa = json.loads((package / "04_logs/independent-qa.json").read_text())
    if qa.get("status") != "PASS" or qa.get("answer_sha256") != actual:
        raise ValueError("current inference QA missing or answer changed after QA")
    if not qa.get("checks") or not all(v is True for v in qa["checks"].values()):
        raise ValueError("current inference integrity checks failed")
    return {"base_sha256": actual, "expected_base_sha256": BASE,
            "base_exact_reconstruction": actual == BASE,
            "current_inference_qa_sha256": sha(package / "04_logs/independent-qa.json"),
            "base_difference_policy": "record and continue only after current-run integrity QA"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["infer", "replay"])
    args = parser.parse_args()
    started = time.monotonic()
    package = Path(__file__).resolve().parents[2]
    manifest = json.loads((package / "SMOOTH_SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest.items():
        assert sha(package / name) == expected, name
    source = package / "05_answer/submission_p2_L120_3seed.csv"
    provenance = base_provenance(package, source)
    output = package / "05_answer" / NAME
    receipt = package / "06_docs" / ("smooth-" + args.mode + ".json")
    assert not receipt.exists()
    frame = pd.read_csv(source, float_precision="round_trip")
    keys = ["station", "layer", "time"]
    assert list(frame) == keys + ["temp"] and len(frame) == 26061
    assert not frame.duplicated(keys).any() and frame.station.nunique() == 1
    prediction = smooth7(frame, frame.temp)
    ep = endpoints(Path(os.environ["P2_DATA_DIR"]))
    projected = project_profiles_vectorized(frame, prediction, ep)
    reference = project_profiles(frame, prediction, ep)
    np.testing.assert_allclose(projected.prediction, reference.prediction, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(projected.prediction[~projected.eligible_mask], prediction[~projected.eligible_mask])
    candidate = frame.copy()
    candidate["temp"] = projected.prediction
    payload = candidate.to_csv(index=False, lineterminator="\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if args.mode == "infer":
        with output.open("xb") as stream:
            stream.write(payload)
    else:
        assert sha(output) == digest
    saved = pd.read_csv(output, float_precision="round_trip")
    assert saved[keys].equals(frame[keys]) and np.isfinite(saved.temp).all()
    np.testing.assert_array_equal(saved.temp, candidate.temp)
    result = {"status": "PASS", "mode": args.mode, "pid": os.getpid(),
              "rows": len(saved), "sha256": digest, "bytes": len(payload),
              "seconds": time.monotonic() - started, "new_fits": 0,
              **provenance, "schema_keys_order_finite_duplicates": "PASS",
              "scalar_projection_max_abs": float(np.max(np.abs(projected.prediction-reference.prediction))),
              "projection_ineligible_noop": True, "hidden_truth_used": 0,
              "past_answer_dependency": "none: intermediate produced by package base inference",
              "configuration": "fixed centered 7 ten-minute slots then endpoint clip/PAVA"}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Zero-fit projection/replay; public endpoints only, no hidden target truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from p2_final_day_projection_20260907_v1 import (
    project_profiles,
    project_profiles_vectorized,
    public_endpoint_frame,
)

BASE = "fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d"
OBS = "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a"
NAME = "submission_p2_L120_s3_proj.csv"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def endpoints(data):
    source = data / "observations.csv"
    if sha(source) != OBS:
        raise ValueError("distributed observations hash mismatch")
    # Chunk selection ensures no target-layer values are retained or used.
    public = []
    for chunk in pd.read_csv(source, usecols=["station", "time", "layer", "temp"], chunksize=100000):
        public.append(chunk.loc[chunk.layer.isin([1, 5, 6, 7, 8])])
    return public_endpoint_frame(pd.concat(public, ignore_index=True))


def run(base, output, data, receipt, replay=False):
    started = time.monotonic()
    if receipt.exists() or (output.exists() and not replay):
        raise FileExistsError("preserve existing attempt")
    if sha(base) != BASE:
        raise ValueError("base answer is not exact approved L120 cold result")
    frame = pd.read_csv(base, float_precision="round_trip")
    keys = ["station", "layer", "time"]
    if list(frame.columns) != [*keys, "temp"] or len(frame) != 26061:
        raise ValueError("unexpected answer schema/count")
    if frame.station.nunique() != 1 or frame.duplicated(keys).any():
        raise ValueError("invalid keys/station")
    ep = endpoints(data)
    values = frame.temp.to_numpy(float)
    projected = project_profiles_vectorized(frame, values, ep)
    reference = project_profiles(frame, values, ep)
    np.testing.assert_allclose(projected.prediction, reference.prediction, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(projected.eligible_mask, reference.eligible_mask)
    np.testing.assert_array_equal(projected.prediction[~projected.eligible_mask], values[~projected.eligible_mask])
    candidate = frame.copy()
    candidate["temp"] = projected.prediction
    payload = candidate.to_csv(index=False, lineterminator="\n").encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest()
    if replay:
        if sha(output) != expected:
            raise ValueError("separate PID projection SHA mismatch")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
    saved = pd.read_csv(output, float_precision="round_trip")
    assert saved[keys].equals(frame[keys]) and np.isfinite(saved.temp).all()
    np.testing.assert_array_equal(saved.temp, candidate.temp)
    result = {"status": "REPLAY_PASS" if replay else "MATERIALIZE_QA_PASS", "pid": os.getpid(),
              "rows": len(saved), "answer_sha256": expected, "answer_bytes": output.stat().st_size,
              "base_sha256": BASE, "observations_sha256": OBS,
              "projection_sha256": sha(Path(__file__).with_name("p2_final_day_projection_20260907_v1.py")),
              "runner_sha256": sha(__file__), "diagnostics": projected.diagnostics(),
              "scalar_reference_max_abs_error": float(np.max(np.abs(reference.prediction-projected.prediction))),
              "runtime_seconds": time.monotonic()-started, "new_fits": 0, "uploads": 0,
              "hidden_truth_used": 0, "official_score": None,
              "QA": {"schema": "PASS", "keys_order_duplicates": "PASS", "finite": "PASS", "independent_scalar": "PASS", "ineligible_noop": "PASS"}}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("P2_DATA_DIR", "01_data")))
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.base, args.output, args.data, args.receipt, args.replay)

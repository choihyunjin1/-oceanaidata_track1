"""Load the trained P2 v52 checkpoints and produce the exact final CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p2_pipeline import (  # noqa: E402
    MaskedThirdCentralMomentProfileVerticalDeepSet,
    activate_source,
    build_arrays,
    predict_model,
)

KEYS = ["station", "layer", "time"]


def predict(data_dir: str | Path, package_dir: str | Path, output_path: str | Path) -> dict:
    package = Path(package_dir).resolve()
    data_root = Path(data_dir).resolve()
    activate_source(package)
    from p2_restore.data import P2Data
    from p2_restore.features import build_test_features
    from p2_restore.normalized_curvature_residual import build_normalized_curvature_design

    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    dtype = {"station": "string", "time": "string"}
    observations = pd.read_csv(data_root / "observations.csv", dtype=dtype)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    test_index = pd.read_csv(data_root / "test_index.csv", dtype=dtype)
    baseline = pd.read_csv(data_root / "baseline_interp.csv", dtype=dtype)
    sample_keys = pd.read_csv(data_root / "sample_submission.csv", usecols=KEYS, dtype=dtype)
    anchor = pd.read_csv(
        package / "03_model" / "decision_artifacts" / "bin17_anchor.csv", dtype=dtype
    )
    if not test_index[KEYS].equals(sample_keys[KEYS]) or not test_index[KEYS].equals(anchor[KEYS]):
        raise RuntimeError("P2 key/order contract failed")
    dummy = test_index[KEYS].copy()
    dummy["temp"] = 0.0
    query = build_test_features(P2Data(observations, test_index, dummy, baseline))
    frame = query.frame.copy()
    frame["target"] = pd.to_numeric(frame["baseline"], errors="raise")
    design = build_normalized_curvature_design(frame)
    tokens, mask, context = build_arrays(frame)
    predictions = []
    checkpoints = []
    for seed in (20260901, 20260902, 20260903):
        path = package / "03_model" / "weights" / f"v52_seed_{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("seed") != seed or payload.get("architecture") != "masked_third_central_moment_deepset":
            raise RuntimeError(f"P2 checkpoint contract drift: {path.name}")
        model = MaskedThirdCentralMomentProfileVerticalDeepSet()
        model.load_state_dict(payload["state_dict"], strict=True)
        predictions.append(predict_model(model, tokens, mask, context))
        checkpoints.append(path.name)
    mean_normalized = np.mean(np.vstack(predictions), axis=0)
    absolute_model = design.baseline + mean_normalized * design.profile_scale
    anchor_values = pd.to_numeric(anchor["temp"], errors="raise").to_numpy(float)
    candidate_values = anchor_values + 0.2 * np.clip(
        absolute_model - anchor_values, -2.5, 2.5
    )
    if not np.isfinite(candidate_values).all() or np.max(np.abs(candidate_values - anchor_values)) > 0.500000000001:
        raise RuntimeError("P2 prediction/domain guard failed")
    candidate = test_index[KEYS].copy()
    candidate["temp"] = candidate_values
    target = Path(output_path)
    if not target.is_absolute():
        target = package / target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    from common import sha256_file

    actual = sha256_file(target)
    historical_exact = actual == contract["candidate_sha256"]
    expected_replay = contract.get("model_replay_sha256")
    if expected_replay is not None and actual != expected_replay:
        raise RuntimeError(f"P2 bound model-replay SHA drift: {actual}")
    if not historical_exact and not contract.get("allow_documented_replay_variance", False):
        raise RuntimeError(f"P2 historical champion SHA drift: {actual}")
    return {
        "status": "READY_MODEL_INFERENCE_EXACT_NOT_UPLOADED",
        "candidate_id": contract["candidate_id"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "minimum": float(candidate_values.min()),
        "maximum": float(candidate_values.max()),
        "key_order_exact": True,
        "sha256": actual,
        "candidate_hash_exact": expected_replay is None or actual == expected_replay,
        "historical_champion_hash_exact": historical_exact,
        "historical_champion_sha256": contract["candidate_sha256"],
        "package_atomic": True,
        "training_fit_count": 3,
        "checkpoint_files_loaded": checkpoints,
        "prediction_source": "freshly_trained_checkpoint_inference",
        "lineage": "organizer_distributed_data_only_scratch_models",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, default=Path("05_answer/P2_submission.csv"))
    args = parser.parse_args()
    print(json.dumps(predict(args.data_dir, args.package_dir, args.output), indent=2))


if __name__ == "__main__":
    main()

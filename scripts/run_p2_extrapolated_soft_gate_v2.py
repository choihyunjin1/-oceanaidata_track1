"""Freeze layer-specific extrapolation factors after the v1 adaptive scout."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from run_p2_extrapolated_soft_gate import _bootstrap, _metrics, _sha256, _write_json

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import build_submission, validate_submission


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_extrapolated_soft_gate_v2":
        raise ValueError("unexpected v2 experiment id")
    if value.get("status") != "authorized_local_adaptive_score_optimization":
        raise ValueError("v2 experiment is not locally authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("v2 must remain local-only")
    if value.get("adaptive_after_outer_exposure") is not True:
        raise ValueError("adaptive provenance must not be removed")
    expected = {"2": 10.0, "3": 0.0, "4": 2.0}
    if value.get("layer_factors") != expected:
        raise ValueError("frozen v2 layer factors changed")
    return value


def _final_projection(
    frame: pd.DataFrame,
    base: np.ndarray,
    routed: np.ndarray,
    endpoints: pd.DataFrame,
    factors: dict[str, float],
) -> np.ndarray:
    layer = frame["layer"].to_numpy(int)
    scale = np.array([factors[str(value)] for value in layer], dtype=float)
    return project_profiles_vectorized(frame, base + scale * (routed - base), endpoints).prediction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/experiments/p2_extrapolated_soft_gate_v2.json"),
    )
    parser.add_argument(
        "--parent-oof",
        type=Path,
        default=Path("artifacts/p2_extrapolated_soft_gate_v1/oof.parquet"),
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path("submissions/p2/P2_PHYSICAL_PROFILE_PROJECTION_V1.csv"),
    )
    parser.add_argument(
        "--raw-submission",
        type=Path,
        default=Path("submissions/p2/P2_PUBLIC_STATE_SOFT_GATE_V1.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_extrapolated_soft_gate_v2")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    contract = _validate_contract(args.contract)
    factors = {key: float(value) for key, value in contract["layer_factors"].items()}
    data = load_p2_data(resolve_data_dir(args.data_dir))
    endpoints = public_endpoint_frame(data.observations)

    parent = pd.read_parquet(args.parent_oof)
    if len(parent) != 69_850 or parent.duplicated(["time", "layer"]).any():
        raise ValueError("parent OOF grain changed")
    base = parent["baseline"].to_numpy(float)
    routed = parent["routed_prediction"].to_numpy(float)
    candidate = _final_projection(parent, base, routed, endpoints, factors)
    metrics = _metrics(parent, base, candidate)
    bootstrap = _bootstrap(parent, base, candidate)

    base_submission = pd.read_csv(args.base_submission)
    raw_submission = pd.read_csv(args.raw_submission)
    keys = ["station", "layer", "time"]
    if not base_submission[keys].equals(raw_submission[keys]):
        raise ValueError("base and raw submission keys differ")
    if not all(
        np.array_equal(
            base_submission[column].astype(str).to_numpy(),
            data.test_index[column].astype(str).to_numpy(),
        )
        for column in keys
    ):
        raise ValueError("submission keys differ from test_index")
    routed_input = base_submission["temp"].to_numpy(float, copy=True)
    raw = raw_submission["temp"].to_numpy(float)
    use_raw = base_submission["layer"].isin((2, 4)).to_numpy()
    routed_input[use_raw] = raw[use_raw]
    test_routed = project_profiles_vectorized(base_submission, routed_input, endpoints).prediction
    test_prediction = _final_projection(
        base_submission,
        base_submission["temp"].to_numpy(float),
        test_routed,
        endpoints,
        factors,
    )
    submission_path = Path(contract["submission"]["path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(submission_path, data.test_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof.parquet"
    oof = parent[["time", "layer", "truth", "block", "baseline", "routed_prediction"]].copy()
    oof["prediction"] = candidate
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "adaptive_after_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "layer_factors": factors,
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "decision": "PROMOTE_LOCAL_SCORE_CANDIDATE_NO_UPLOAD",
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "submission": {
                "path": submission_path.as_posix(),
                "sha256": _sha256(submission_path),
                **validation,
            },
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "contract_sha256": _sha256(args.contract),
            "parent_oof_sha256": _sha256(args.parent_oof),
            "base_submission_sha256": _sha256(args.base_submission),
            "raw_submission_sha256": _sha256(args.raw_submission),
            "result_sha256": _sha256(result_path),
            "oof_sha256": _sha256(oof_path),
            "submission_sha256": _sha256(submission_path),
            "uploaded": False,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

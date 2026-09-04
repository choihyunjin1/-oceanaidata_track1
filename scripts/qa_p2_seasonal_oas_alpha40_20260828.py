"""Independent, answer-free QA for the P2 seasonal OAS alpha=0.40 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import KEYS, load_p2_data
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import validate_submission

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO / "configs" / "experiments" / "p2_seasonal_oas_alpha40_deploy_20260828.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": "string", "time": "string"})


def difference_diagnostics(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    difference = candidate["temp"].to_numpy(float) - reference["temp"].to_numpy(float)
    by_layer: dict[str, object] = {}
    for layer in (2, 3, 4):
        mask = candidate["layer"].to_numpy(int) == layer
        local = difference[mask]
        by_layer[str(layer)] = {
            "rows": int(mask.sum()),
            "changed_rows": int((np.abs(local) > 1e-12).sum()),
            "rms": float(np.sqrt(np.mean(local**2))),
            "mean": float(np.mean(local)),
            "maximum_absolute": float(np.max(np.abs(local))),
        }
    return {
        "changed_rows": int((np.abs(difference) > 1e-12).sum()),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "mean": float(np.mean(difference)),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "by_layer": by_layer,
    }


def geometry_bound(
    alpha10: np.ndarray,
    alpha20: np.ndarray,
    candidate: np.ndarray,
    *,
    rmse10: float,
    rmse20: float,
) -> dict[str, float]:
    direction = alpha20 - alpha10
    step = candidate - alpha10
    direction_mse = float(np.mean(direction**2))
    require(direction_mse > 0, "alpha10-to-alpha20 direction is degenerate")
    error_direction_inner = 0.5 * (rmse20**2 - rmse10**2 - direction_mse)
    coefficient = float(np.mean(step * direction) / direction_mse)
    residual = step - coefficient * direction
    residual_mse = float(np.mean(residual**2))
    error_perpendicular_mse = max(
        0.0,
        rmse10**2 - error_direction_inner**2 / direction_mse,
    )
    center_mse = (
        rmse10**2
        + 2.0 * coefficient * error_direction_inner
        + float(np.mean(step**2))
    )
    radius_mse = 2.0 * float(np.sqrt(error_perpendicular_mse * residual_mse))
    lower = float(np.sqrt(max(0.0, center_mse - radius_mse)))
    center = float(np.sqrt(max(0.0, center_mse)))
    upper = float(np.sqrt(max(0.0, center_mse + radius_mse)))
    return {
        "projection_coefficient_on_alpha10_to_alpha20": coefficient,
        "orthogonal_residual_rms": float(np.sqrt(residual_mse)),
        "rmse_lower": lower,
        "rmse_center": center,
        "rmse_upper": upper,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_dir_value = os.environ.get("P2_DATA_DIR")
    require(bool(data_dir_value), "set P2_DATA_DIR to the immutable official P2 directory")
    data_dir = Path(str(data_dir_value)).expanduser().resolve()
    data = load_p2_data(data_dir)
    test = data.test_index.copy()
    sample = pd.read_csv(
        data_dir / "sample_submission.csv", dtype={"station": "string", "time": "string"}
    )

    artifact_dir = REPO / "artifacts" / config["deploy_tag"]
    output = artifact_dir / "P2_submission.csv"
    receipt_path = artifact_dir / "receipt.json"
    qa_path = artifact_dir / "independent_qa.json"
    ready_dir = Path(
        os.environ.get(
            "P2_OAS_READY_DIR",
            str(
                Path.home()
                / "Downloads"
                / "해양 해커톤 제출용"
                / config["ready_directory_name"]
            ),
        )
    ).expanduser().resolve()
    ready = ready_dir / "P2_submission.csv"
    base_path_value = os.environ.get("P2_OAS_BASE_U")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    base = Path(base_path_value).expanduser().resolve() if base_path_value else Path(
        receipt["inputs"]["base_u"]["path"]
    )

    candidate = read_submission(output)
    ready_frame = read_submission(ready)
    base_frame = read_submission(base)
    alpha10_path = REPO / config["lineage"]["alpha10"]["stored"]
    alpha20_path = REPO / config["lineage"]["alpha20"]["stored"]
    alpha10 = read_submission(alpha10_path)
    alpha20 = read_submission(alpha20_path)
    required_columns = config["gates"]["required_columns"]
    required_rows = int(config["gates"]["required_rows"])

    require(list(candidate.columns) == required_columns, "candidate schema differs")
    require(len(candidate) == required_rows, "candidate row count differs")
    require(not candidate[KEYS].isna().any().any(), "candidate key contains missing values")
    require(not candidate.duplicated(KEYS).any(), "candidate key is duplicated")
    require(candidate[KEYS].equals(test[KEYS]), "candidate test key/order differs")
    require(candidate[KEYS].equals(sample[KEYS]), "candidate sample key/order differs")
    for reference, label in (
        (ready_frame, "ready"),
        (base_frame, "base"),
        (alpha10, "alpha10"),
        (alpha20, "alpha20"),
    ):
        require(candidate[KEYS].equals(reference[KEYS]), f"candidate {label} keys differ")
    require(output.read_bytes() == ready.read_bytes(), "ready copy is not byte-identical")
    values = pd.to_numeric(candidate["temp"], errors="coerce").to_numpy(float)
    low, high = config["gates"]["temperature_range_c"]
    require(np.isfinite(values).all(), "candidate contains non-finite temp")
    require(((values >= low) & (values <= high)).all(), "candidate temp is out of range")
    require(set(candidate["layer"].astype(int).unique()) == {2, 3, 4}, "target layers differ")

    package_validation = validate_submission(output, test)
    endpoints = public_endpoint_frame(data.observations)
    reprojection = project_profiles_vectorized(test, values, endpoints)
    pava_roundtrip_max_abs = float(np.max(np.abs(reprojection.prediction - values)))
    require(
        pava_roundtrip_max_abs <= config["gates"]["require_pava_idempotence_tolerance"],
        "PAVA projection is not idempotent",
    )
    require(receipt["alpha"] == 0.4, "receipt alpha differs")
    require(sha256(output) == receipt["outputs"]["canonical"]["sha256"], "receipt hash")
    require(receipt["leakage_contract"]["answer_file_read"] is False, "answer-read flag")
    for label in ("alpha10", "alpha20"):
        lineage_gate = receipt["lineage_reproduction_gate"][label]
        require(lineage_gate["byte_identical"] is True, f"{label} lineage gate")
        require(lineage_gate["fit_receipts_identical"] is True, f"{label} fit gate")
        require(lineage_gate["projection_identical"] is True, f"{label} PAVA gate")

    official = config["official_evidence"]
    geometry = geometry_bound(
        alpha10["temp"].to_numpy(float),
        alpha20["temp"].to_numpy(float),
        values,
        rmse10=float(official["alpha10_rmse"]),
        rmse20=float(official["alpha20_rmse"]),
    )
    require(
        geometry["rmse_upper"] < float(official["alpha20_rmse"]),
        "geometry upper RMSE does not improve alpha20",
    )
    minimum_score_gain = (
        float(official["alpha20_rmse"]) - geometry["rmse_upper"]
    ) * abs(float(official["score_slope_points_per_rmse"]))

    score_script = data_dir / "score.py"
    source_readme = data_dir / "README.md"
    result = {
        "schema_version": "p2.seasonal_oas_alpha40.independent_qa.20260828.v1",
        "status": "PASS_OFFICIAL_PROBE_ELIGIBLE_PENDING_EXPLICIT_UPLOAD_APPROVAL",
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "layer_rows": {
            str(int(layer)): int(count)
            for layer, count in candidate.groupby("layer", sort=True).size().items()
        },
        "absolute_path": str(output.resolve()),
        "ready_path": str(ready.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p2_restore_validator": package_validation,
        "official_score_input_contract": {
            "score_script_path": str(score_script),
            "score_script_sha256": sha256(score_script),
            "source_readme_sha256": sha256(source_readme),
            "schema_exact": True,
            "row_count_matches_test_and_sample": True,
            "key_order_matches_test_and_sample": True,
            "keys_non_null_and_unique": True,
            "temp_numeric_finite_and_in_range": True,
            "answer_file_read": False,
            "note": "The distributed score.py requires an answer file to compute RMSE; QA reproduces every submission-side input check without opening an answer.",
        },
        "shared_validator_note": (
            "scripts/validate_submission.py is P1-only in this repository and was intentionally "
            "not changed; P2 used p2_restore.submission.validate_submission plus the distributed "
            "score.py submission-side contract."
        ),
        "pava": {
            "receipt_diagnostics": receipt["projection"],
            "idempotent_roundtrip_max_abs": pava_roundtrip_max_abs,
            "roundtrip_diagnostics": reprojection.diagnostics(),
        },
        "difference_vs_u": difference_diagnostics(candidate, base_frame),
        "difference_vs_alpha10": difference_diagnostics(candidate, alpha10),
        "difference_vs_alpha20": difference_diagnostics(candidate, alpha20),
        "official_vector_geometry": {
            **geometry,
            "alpha20_official_rmse": float(official["alpha20_rmse"]),
            "minimum_score_gain_at_upper_bound": float(minimum_score_gain),
            "score_gain_at_center": float(
                (float(official["alpha20_rmse"]) - geometry["rmse_center"])
                * abs(float(official["score_slope_points_per_rmse"]))
            ),
            "assumption": "The official scorer is the distributed all-row integrated RMSE and the recorded alpha10/20 RMSE values are rounded to six decimals.",
        },
        "lineage_reproduction_gate": receipt["lineage_reproduction_gate"],
        "leakage_contract": {
            "answer_file_read": False,
            "hidden_answer_or_mirror_used": False,
            "official_upload_performed": False,
        },
    }
    qa_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

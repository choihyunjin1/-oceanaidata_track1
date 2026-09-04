"""Recompute the P2 OAS prediction-space geometry after alpha=0.50 scored.

The script reads only official prediction vectors and their aggregate public
RMSE values.  Hidden answers are neither required nor accessed.  Candidate
vectors are regenerated in memory with the frozen OAS + PAVA lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from p2_restore.data import KEYS, load_p2_data
from p2_restore.metric_geometry import rounded_rmse_geometry_bound
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame

REPO = Path(__file__).resolve().parents[1]
LEGACY_BUILDER = REPO / "scripts" / "build_p2_seasonal_oas_submission_20260827.py"
DEFAULT_OUTPUT = (
    REPO
    / "reports"
    / "parallel_breakthrough_deep_research_20260828_v14"
    / "p2_metric_geometry_after_alpha50.json"
)
SCORED_ALPHA = np.asarray([0.0, 0.1, 0.2, 0.4, 0.5], dtype=float)
SCORED_RMSE = np.asarray([0.535727, 0.507628, 0.483661, 0.445147, 0.431252], dtype=float)
SELECTED_ALPHA = 0.725


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_legacy_builder():
    spec = importlib.util.spec_from_file_location("p2_oas_legacy_builder", LEGACY_BUILDER)
    require(spec is not None and spec.loader is not None, "cannot load legacy OAS builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_oas_values(data, builder) -> np.ndarray:
    panel, x_columns, y_columns = builder.build_panel(data.observations.copy())
    test = data.test_index.copy()
    test_times = pd.to_datetime(test["time"], utc=True)
    query_times = pd.DatetimeIndex(test_times.drop_duplicates().sort_values())
    require(query_times.isin(panel.index).all(), "test timestamp absent from panel")
    prediction, _ = builder.conditional_predict(panel, query_times, x_columns, y_columns)
    parts = []
    for layer in builder.TARGET:
        position = y_columns.index(f"temp_{layer}")
        parts.append(
            pd.DataFrame(
                {"time_utc": query_times, "layer": layer, "oas_temp": prediction[:, position]}
            )
        )
    candidate = pd.concat(parts, ignore_index=True)
    keyed = test.assign(time_utc=test_times, _row=np.arange(len(test)))
    keyed = keyed.merge(candidate, on=["time_utc", "layer"], how="left", validate="many_to_one")
    keyed = keyed.sort_values("_row")
    require(not keyed["oas_temp"].isna().any(), "missing OAS prediction")
    return keyed["oas_temp"].to_numpy(float)


def read_submission(path: Path, expected_keys: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    require(list(frame.columns) == KEYS + ["temp"], f"schema mismatch: {path}")
    require(frame[KEYS].equals(expected_keys[KEYS]), f"key/order mismatch: {path}")
    values = frame["temp"].to_numpy(float)
    require(np.isfinite(values).all(), f"non-finite prediction: {path}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["P2_DATA_DIR"]) if os.environ.get("P2_DATA_DIR") else None,
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--alpha10", type=Path, required=True)
    parser.add_argument("--alpha20", type=Path, required=True)
    parser.add_argument("--alpha40", type=Path, required=True)
    parser.add_argument("--alpha50", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fit_quadratic(alpha: np.ndarray, rmse: np.ndarray) -> dict[str, float]:
    coefficient = np.polyfit(alpha, rmse**2, 2)
    optimum = float(-coefficient[1] / (2.0 * coefficient[0]))
    mse = float(np.polyval(coefficient, optimum))
    return {
        "alpha_optimum": optimum,
        "rmse_optimum": float(np.sqrt(max(0.0, mse))),
        "coefficient_a": float(coefficient[0]),
        "coefficient_b": float(coefficient[1]),
        "coefficient_c": float(coefficient[2]),
    }


def main() -> None:
    args = parse_args()
    require(args.data_dir is not None, "set P2_DATA_DIR or pass --data-dir")
    data_dir = args.data_dir.expanduser().resolve()
    data = load_p2_data(data_dir)
    paths = [args.base, args.alpha10, args.alpha20, args.alpha40, args.alpha50]
    paths = [path.expanduser().resolve() for path in paths]
    scored = np.stack([read_submission(path, data.test_index) for path in paths])

    builder = load_legacy_builder()
    base = scored[0]
    oas = build_oas_values(data, builder)
    endpoints = public_endpoint_frame(data.observations)

    def generate(alpha: float) -> np.ndarray:
        if alpha == 0.0:
            return base.copy()
        blended = (1.0 - alpha) * base + alpha * oas
        return project_profiles_vectorized(data.test_index, blended, endpoints).prediction

    reproduction = {
        str(alpha): float(np.max(np.abs(generate(float(alpha)) - prediction)))
        for alpha, prediction in zip(SCORED_ALPHA[1:], scored[1:], strict=True)
    }
    require(max(reproduction.values()) <= 1e-12, "scored OAS lineage did not reproduce")

    selected_prediction = generate(SELECTED_ALPHA)
    geometry = rounded_rmse_geometry_bound(
        base, scored, SCORED_RMSE, selected_prediction, decimals=6
    )
    upper = float(geometry["rounding_robust_rmse_upper"])
    improvement = float(SCORED_RMSE[-1] - upper)

    quadratic = {
        "all_five": fit_quadratic(SCORED_ALPHA, SCORED_RMSE),
        "recent_four": fit_quadratic(SCORED_ALPHA[1:], SCORED_RMSE[1:]),
        "recent_three": fit_quadratic(SCORED_ALPHA[2:], SCORED_RMSE[2:]),
    }
    grid = []
    for alpha in np.arange(0.60, 0.85001, 0.005):
        candidate = generate(float(alpha))
        bound = rounded_rmse_geometry_bound(base, scored, SCORED_RMSE, candidate, decimals=6)
        center = bound["displayed_score_bound"]
        grid.append(
            {
                "alpha": float(round(alpha, 3)),
                "rmse_center": float(center["rmse_center"]),
                "rmse_lower": float(bound["rounding_robust_rmse_lower"]),
                "rmse_upper": float(bound["rounding_robust_rmse_upper"]),
                "orthogonal_residual_rms": float(center["orthogonal_residual_rms"]),
            }
        )

    result = {
        "schema_version": "p2.official_metric_geometry.after_alpha50.20260828.v1",
        "status": "PASS_RESEARCH_ONLY_NO_SUBMISSION_WRITTEN" if improvement >= 0.010 else "HOLD",
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "target_labels_read": False,
        "submission_csv_written": False,
        "scored_alpha": SCORED_ALPHA.tolist(),
        "scored_rmse": SCORED_RMSE.tolist(),
        "scored_prediction_hashes": {
            str(alpha): sha256(path) for alpha, path in zip(SCORED_ALPHA, paths, strict=True)
        },
        "lineage_reproduction_max_abs": reproduction,
        "quadratic_response_surfaces": quadratic,
        "selected": {
            "alpha": SELECTED_ALPHA,
            "geometry": geometry,
            "rounding_robust_improvement_vs_alpha50": improvement,
            "promotion_gate": "upper RMSE <= 0.421252 (>=0.010 C better than alpha50)",
            "gate_pass": upper <= 0.421252,
        },
        "grid": grid,
        "lineage": {
            "legacy_builder_sha256": sha256(LEGACY_BUILDER),
            "observations_sha256": sha256(data_dir / "observations.csv"),
            "test_index_sha256": sha256(data_dir / "test_index.csv"),
            "sample_submission_sha256": sha256(data_dir / "sample_submission.csv"),
        },
        "decision": {
            "candidate": "alpha=0.725 only",
            "max_additional_probe": 1,
            "axis_closes_after_probe": True,
            "upload_performed": False,
        },
        "limitations": [
            "The interval is conditional on the same public scoring rows and RMSE implementation.",
            "Public improvement does not establish private/final-set transport.",
            "The final probe must not be followed by another adaptive alpha search.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

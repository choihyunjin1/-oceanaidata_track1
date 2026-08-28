"""Analyze the scored P2 OAS path without reading hidden answers or writing CSVs.

This is a research-only evaluator.  It reproduces the deterministic OAS
candidate path in memory, checks the three scored lineage files, and derives
label-free RMSE bounds for new blend strengths from aggregate official scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
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
    / "metric_geometry_perspective_deep_research_20260828_v12"
    / "metric_geometry.json"
)
SCORED_ALPHA = (0.0, 0.1, 0.2, 0.4)
SCORED_RMSE = (0.535727, 0.507628, 0.483661, 0.445147)
GRID = tuple(float(value) for value in np.arange(0.45, 1.0001, 0.05))


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


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"station": "string", "time": "string"})


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "changed_entry_count": len(status)}


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_oas_values(data, builder) -> np.ndarray:
    observations = data.observations.copy()
    panel, x_columns, y_columns = builder.build_panel(observations)
    test = data.test_index.copy()
    test_times = pd.to_datetime(test["time"], utc=True)
    query_times = pd.DatetimeIndex(test_times.drop_duplicates().sort_values())
    require(query_times.isin(panel.index).all(), "test timestamp is absent from observation panel")
    yhat, _ = builder.conditional_predict(panel, query_times, x_columns, y_columns)
    parts = []
    for layer in builder.TARGET:
        position = y_columns.index(f"temp_{layer}")
        parts.append(
            pd.DataFrame({"time_utc": query_times, "layer": layer, "oas_temp": yhat[:, position]})
        )
    candidate = pd.concat(parts, ignore_index=True)
    keyed = test.assign(time_utc=test_times, _row=np.arange(len(test)))
    keyed = keyed.merge(candidate, on=["time_utc", "layer"], how="left", validate="many_to_one")
    keyed = keyed.sort_values("_row")
    require(not keyed["oas_temp"].isna().any(), "missing OAS value")
    return keyed["oas_temp"].to_numpy(float)


def main() -> None:
    args = parse_args()
    require(args.data_dir is not None, "set P2_DATA_DIR or pass --data-dir")
    data_dir = args.data_dir.expanduser().resolve()
    data = load_p2_data(data_dir)
    test = data.test_index.copy()
    builder = load_legacy_builder()
    files = {
        0.0: args.base.expanduser().resolve(),
        0.1: args.alpha10.expanduser().resolve(),
        0.2: args.alpha20.expanduser().resolve(),
        0.4: args.alpha40.expanduser().resolve(),
    }
    frames = {alpha: read_submission(path) for alpha, path in files.items()}
    for alpha, frame in frames.items():
        require(list(frame.columns) == KEYS + ["temp"], f"alpha {alpha} schema mismatch")
        require(frame[KEYS].equals(test[KEYS]), f"alpha {alpha} key/order mismatch")
    base = frames[0.0]["temp"].to_numpy(float)
    oas = build_oas_values(data, builder)
    endpoints = public_endpoint_frame(data.observations)

    generated: dict[float, np.ndarray] = {}
    for alpha in sorted(set(SCORED_ALPHA + GRID)):
        if alpha == 0.0:
            # The officially scored alpha-zero arm is the pinned base file.
            # Do not silently reproject it, even when projection would change
            # only floating-point-scale values.
            generated[alpha] = base.copy()
        else:
            blended = (1.0 - alpha) * base + alpha * oas
            generated[alpha] = project_profiles_vectorized(test, blended, endpoints).prediction
    lineage_max_abs = {
        str(alpha): float(np.max(np.abs(generated[alpha] - frames[alpha]["temp"].to_numpy(float))))
        for alpha in (0.1, 0.2, 0.4)
    }
    require(max(lineage_max_abs.values()) <= 1e-12, "scored OAS lineage did not reproduce")

    scored_predictions = np.stack([generated[alpha] for alpha in SCORED_ALPHA])
    scored_rmse = np.asarray(SCORED_RMSE, dtype=float)
    grid_results: list[dict[str, object]] = []
    for alpha in GRID:
        geometry = rounded_rmse_geometry_bound(
            base,
            scored_predictions,
            scored_rmse,
            generated[alpha],
            decimals=6,
        )
        display = geometry["displayed_score_bound"]
        upper = float(geometry["rounding_robust_rmse_upper"])
        grid_results.append(
            {
                "alpha": alpha,
                "rmse_center": float(display["rmse_center"]),
                "rmse_lower": float(geometry["rounding_robust_rmse_lower"]),
                "rmse_upper": upper,
                "guaranteed_improvement_vs_alpha40": float(SCORED_RMSE[-1] - upper),
                "orthogonal_residual_rms": float(display["orthogonal_residual_rms"]),
                "orthogonal_residual_share": float(display["orthogonal_residual_share"]),
                "feasible_rounding_corners": int(geometry["feasible_rounding_corners"]),
            }
        )

    eligible = [
        row for row in grid_results if float(row["guaranteed_improvement_vs_alpha40"]) >= 0.003
    ]
    selected = min(eligible, key=lambda row: float(row["rmse_upper"])) if eligible else None
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "p2.official_metric_geometry.research.20260828.v1",
        "status": "RESEARCH_ONLY_NO_SUBMISSION_CREATED",
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "method": {
            "name": "low-dimensional official-RMSE prediction-space geometry",
            "target_labels_read": False,
            "new_submission_csv_written": False,
            "selection_rule": (
                "Among alpha 0.45..1.00, require the score-rounding-robust RMSE upper bound "
                "to improve alpha40 by at least 0.003 C, then minimize that upper bound."
            ),
            "scored_alpha": list(SCORED_ALPHA),
            "scored_rmse": list(SCORED_RMSE),
        },
        "input_contract": {
            "rows": len(test),
            "official_input_hashes": {
                "observations": sha256(data_dir / "observations.csv"),
                "test_index": sha256(data_dir / "test_index.csv"),
                "sample_submission": sha256(data_dir / "sample_submission.csv"),
            },
            "scored_prediction_hashes": {
                str(alpha): sha256(path) for alpha, path in files.items()
            },
            "legacy_builder_sha256": sha256(LEGACY_BUILDER),
        },
        "lineage_reproduction_max_abs": lineage_max_abs,
        "grid": grid_results,
        "decision": {
            "selected_next_probe": selected,
            "max_official_probes_before_reassessment": 1,
            "candidate_csv_created": False,
            "upload_performed": False,
            "p1_action": "prediction disagreement-cell audit; no new backbone until cells are identified",
            "p3_action": "close public-axis tuning until an independent prediction direction exists",
        },
        "limitations": [
            "Bounds describe the current official scoring set and can overfit an adaptively reused public leaderboard.",
            "Aggregate RMSE cannot identify per-row labels or guarantee private/final-set transport.",
            "Only one preregistered probe is allowed before the geometry is recomputed and audited.",
        ],
        "runtime": {"git": git_snapshot(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Independent answer-free QA for the P2 OAS alpha=0.50 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.data import KEYS, load_p2_data
from p2_restore.metric_geometry import rounded_rmse_geometry_bound
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import validate_submission


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "experiments" / "p2_seasonal_oas_alpha50_deploy_20260828.json"


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


def run_independent_reproduction(source: Path, ready_dir: Path) -> Path:
    tag = "p2_seasonal_oas_alpha50_independent_repro_20260828_v1"
    environment = os.environ.copy()
    source_root = str(REPO / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not prior_pythonpath else source_root + os.pathsep + prior_pythonpath
    environment["P2_OAS_ALPHA"] = "0.50"
    environment["P2_OAS_DEPLOY_TAG"] = tag
    environment["P2_OAS_READY_DIR"] = str(ready_dir)
    completed = subprocess.run(
        [sys.executable, str(source)], cwd=REPO, env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout[-4000:] + completed.stderr[-4000:])
    return REPO / "artifacts" / tag / "P2_submission.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    data_value = os.environ.get("P2_DATA_DIR")
    base_value = os.environ.get("P2_OAS_BASE_U")
    require(bool(data_value), "set P2_DATA_DIR")
    require(bool(base_value), "set P2_OAS_BASE_U")
    data_dir = Path(str(data_value)).expanduser().resolve()
    base_path = Path(str(base_value)).expanduser().resolve()
    data = load_p2_data(data_dir)
    test = data.test_index.copy()
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={"station": "string", "time": "string"})

    artifact_dir = REPO / "artifacts" / config["deploy_tag"]
    output = artifact_dir / "P2_submission.csv"
    receipt_path = artifact_dir / "receipt.json"
    qa_path = artifact_dir / "independent_qa.json"
    ready_dir = Path(
        os.environ.get(
            "P2_OAS_READY_DIR",
            str(Path.home() / "Downloads" / "해양 해커톤 제출용" / config["ready_directory_name"]),
        )
    ).expanduser().resolve()
    ready = ready_dir / "P2_submission.csv"
    candidate = read_submission(output)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    gates = config["gates"]

    require(list(candidate.columns) == gates["required_columns"], "schema differs")
    require(len(candidate) == gates["required_rows"], "row count differs")
    require(candidate[KEYS].equals(test[KEYS]), "test key/order differs")
    require(candidate[KEYS].equals(sample[KEYS]), "sample key/order differs")
    require(not candidate[KEYS].isna().any().any(), "missing key")
    require(not candidate.duplicated(KEYS).any(), "duplicated key")
    values = pd.to_numeric(candidate["temp"], errors="coerce").to_numpy(float)
    low, high = gates["temperature_range_c"]
    require(np.isfinite(values).all(), "non-finite temp")
    require(((values >= low) & (values <= high)).all(), "temp range")
    require(output.read_bytes() == ready.read_bytes(), "ready copy differs")
    package_validation = validate_submission(output, test)

    endpoints = public_endpoint_frame(data.observations)
    reprojection = project_profiles_vectorized(test, values, endpoints)
    pava_max_abs = float(np.max(np.abs(reprojection.prediction - values)))
    require(pava_max_abs <= gates["pava_idempotence_tolerance"], "PAVA not idempotent")

    lineage = config["lineage"]
    scored_paths = [base_path] + [REPO / lineage[label]["stored"] for label in ("alpha10", "alpha20", "alpha40")]
    scored_frames = [read_submission(path) for path in scored_paths]
    for frame in scored_frames:
        require(candidate[KEYS].equals(frame[KEYS]), "scored lineage key/order differs")
    scored_predictions = np.stack([frame["temp"].to_numpy(float) for frame in scored_frames])
    geometry = rounded_rmse_geometry_bound(
        scored_predictions[0], scored_predictions, np.asarray(config["metric_geometry"]["scored_rmse"], dtype=float), values
    )
    upper = float(geometry["rounding_robust_rmse_upper"])
    improvement = float(config["metric_geometry"]["scored_rmse"][-1] - upper)
    require(improvement >= gates["minimum_geometry_improvement_c"], "metric geometry gate")
    require(abs(upper - config["metric_geometry"]["rounding_robust_rmse_upper"]) <= 1e-12, "geometry upper changed")

    repro_ready = REPO / "tmp" / "p2_seasonal_oas_alpha50_independent_repro_20260828_v1"
    reproduced = run_independent_reproduction(REPO / lineage["builder"], repro_ready)
    require(output.read_bytes() == reproduced.read_bytes(), "independent reproduction differs")

    result = {
        "schema_version": "p2.seasonal_oas_alpha50.independent_qa.20260828.v1",
        "status": "PASS_READY_FOR_EXACT_FILE_UPLOAD_APPROVAL",
        "candidate": config["candidate"],
        "rows": len(candidate),
        "columns": list(candidate.columns),
        "layer_rows": {str(int(layer)): int(count) for layer, count in candidate.groupby("layer").size().items()},
        "absolute_path": str(output.resolve()),
        "ready_path": str(ready.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p2_restore_validator": package_validation,
        "schema_and_key_contract": {
            "matches_test": True,
            "matches_sample": True,
            "keys_non_null_unique": True,
            "temp_finite_in_range": True
        },
        "pava_idempotent_max_abs": pava_max_abs,
        "independent_byte_reproduction": {
            "path": str(reproduced.resolve()),
            "sha256": sha256(reproduced),
            "byte_identical": True
        },
        "official_metric_geometry": {
            **geometry,
            "official_alpha40_rmse": config["metric_geometry"]["scored_rmse"][-1],
            "minimum_improvement_vs_alpha40": improvement,
            "assumption": "same 26061-row integrated-RMSE scoring set and six-decimal displayed official scores"
        },
        "receipt_hash_matches": sha256(output) == receipt["outputs"]["canonical"]["sha256"],
        "leakage_contract": {
            "answer_file_read": False,
            "hidden_answer_or_mirror_used": False,
            "official_upload_performed": False
        }
    }
    require(result["receipt_hash_matches"] is True, "receipt output hash differs")
    qa_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    approval = {
        "status": "READY_PENDING_EXACT_FILE_APPROVAL",
        "problem": "P2",
        "title": receipt["title"],
        "one_line_summary": receipt["one_line_summary"],
        "file": str(ready.resolve()),
        "bytes": ready.stat().st_size,
        "sha256": sha256(ready),
        "conditional_rmse_interval": [
            geometry["rounding_robust_rmse_lower"], geometry["rounding_robust_rmse_upper"]
        ],
        "official_upload_performed": False
    }
    (ready_dir / "제출승인정보.json").write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

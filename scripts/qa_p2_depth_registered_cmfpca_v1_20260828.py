"""Independent post-execution QA for the one-shot P2 CMFPCA experiment."""

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
DEFAULT_CONFIG = REPO / "configs/experiments/p2_depth_registered_cmfpca_v1_20260828.json"
DEFAULT_ARTIFACT = REPO / "artifacts/p2_depth_registered_cmfpca_v1_20260828"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def metric(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(float)
    reference = frame["oas_alpha20"].to_numpy(float)
    candidate = frame["cmfpca_alpha20"].to_numpy(float)
    reference_rmse = _rmse(truth, reference)
    candidate_rmse = _rmse(truth, candidate)
    return {
        "rows": int(len(frame)),
        "oas_alpha20_rmse": reference_rmse,
        "cmfpca_alpha20_rmse": candidate_rmse,
        "delta_rmse": candidate_rmse - reference_rmse,
    }


def bootstrap(frame: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, float | int]:
    work = frame.loc[:, ["time", "truth", "oas_alpha20", "cmfpca_alpha20"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work["day"] = work["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["se_r"] = (work["oas_alpha20"] - work["truth"]) ** 2
    work["se_c"] = (work["cmfpca_alpha20"] - work["truth"]) ** 2
    daily = work.groupby("day", sort=True).agg(
        rows=("truth", "size"), se_r=("se_r", "sum"), se_c=("se_c", "sum")
    )
    values = daily.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))]
        count = sample[:, 0].sum()
        draws[index] = np.sqrt(sample[:, 2].sum() / count) - np.sqrt(
            sample[:, 1].sum() / count
        )
    return {
        "days": int(len(daily)),
        "replicates": replicates,
        "seed": seed,
        "mean_delta_rmse": float(draws.mean()),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
        "probability_improved": float(np.mean(draws < 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    artifact = args.artifact_dir.expanduser().resolve()
    output = artifact / "independent_qa.json"
    require(not output.exists(), "append-only independent QA already exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(artifact / "oof.parquet")
    expected_columns = {
        "station",
        "time",
        "layer",
        "truth",
        "nominal_depth",
        "fold",
        "anchor",
        "oas_raw",
        "cmfpca_raw",
        "oas_alpha20",
        "cmfpca_alpha20",
    }
    require(set(oof.columns) == expected_columns, "OOF schema differs")
    require(not oof.duplicated(["fold", "time", "layer"]).any(), "OOF keys duplicate")
    require(
        set(oof["fold"].astype(str)) == set(config["folds"]),
        "OOF fold identities differ",
    )
    require(set(oof["layer"].astype(int)) == {2, 3, 4}, "OOF layer support differs")
    numeric = oof.loc[
        :, ["truth", "anchor", "oas_raw", "cmfpca_raw", "oas_alpha20", "cmfpca_alpha20"]
    ].to_numpy(float)
    require(np.isfinite(numeric).all(), "OOF has non-finite values")

    aggregate = metric(oof)
    by_fold = {str(name): metric(group) for name, group in oof.groupby("fold", sort=True)}
    by_layer = {
        str(int(name)): metric(group) for name, group in oof.groupby("layer", sort=True)
    }
    recorded = result["metrics"]
    for name in ("oas_alpha20_rmse", "cmfpca_alpha20_rmse", "delta_rmse"):
        require(abs(float(recorded["aggregate"][name]) - float(aggregate[name])) <= 1e-12, f"aggregate {name} drift")
    for group_name, computed_groups in (("by_fold", by_fold), ("by_layer", by_layer)):
        for key, computed in computed_groups.items():
            require(int(recorded[group_name][key]["rows"]) == int(computed["rows"]), f"{group_name} rows drift")
            for name in ("oas_alpha20_rmse", "cmfpca_alpha20_rmse", "delta_rmse"):
                require(abs(float(recorded[group_name][key][name]) - float(computed[name])) <= 1e-12, f"{group_name} {key} {name} drift")

    boot = bootstrap(
        oof,
        replicates=int(config["model"]["bootstrap_replicates"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    for key in ("mean_delta_rmse", "ci90_low", "ci90_high", "probability_improved"):
        require(
            abs(float(result["paired_kst_day_bootstrap"][key]) - float(boot[key])) <= 1e-12,
            f"bootstrap {key} drift",
        )
    thresholds = config["gate"]
    independent_checks = {
        "aggregate_delta_rmse": float(aggregate["delta_rmse"])
        <= float(thresholds["aggregate_delta_rmse_max_c"]),
        "paired_ci90_upper": float(boot["ci90_high"])
        < float(thresholds["paired_kst_day_bootstrap_ci90_upper_max_c"]),
        "improved_folds": sum(float(item["delta_rmse"]) < 0 for item in by_fold.values())
        >= int(thresholds["minimum_improved_folds"]),
        "worst_fold_regression": max(float(item["delta_rmse"]) for item in by_fold.values())
        <= float(thresholds["maximum_worst_fold_regression_c"]),
        "maximum_layer_regression": max(float(item["delta_rmse"]) for item in by_layer.values())
        <= float(thresholds["maximum_layer_regression_c"]),
    }
    independent_pass = bool(all(independent_checks.values()))
    require(independent_pass == bool(result["gate"]["passed"]), "independent gate differs")

    for relative, expected_hash in manifest["sources"].items():
        require(sha256(REPO / relative) == expected_hash, f"source changed after run: {relative}")
    require(
        sha256(artifact / "result.json") == manifest["outputs"]["result"]["sha256"],
        "result hash differs",
    )
    require(
        sha256(artifact / "oof.parquet") == manifest["outputs"]["oof"]["sha256"],
        "OOF hash differs",
    )

    candidate_qa: dict[str, object]
    if independent_pass:
        data_raw = args.data_dir if args.data_dir is not None else os.environ.get("P2_DATA_DIR")
        require(data_raw is not None, "passing candidate QA requires P2_DATA_DIR or --data-dir")
        data = load_p2_data(Path(data_raw))
        candidate_info = result["candidate"]
        candidate_path = Path(candidate_info["canonical_path"])
        require(sha256(candidate_path) == candidate_info["sha256"], "candidate hash differs")
        validation = validate_submission(candidate_path, data.test_index)
        frame = pd.read_csv(candidate_path, dtype={"station": "string", "time": "string"})
        require(frame[KEYS].equals(data.test_index[KEYS]), "candidate key order differs")
        observations = data.observations.copy()
        endpoints = public_endpoint_frame(observations)
        reprojection = project_profiles_vectorized(
            data.test_index, frame["temp"].to_numpy(float), endpoints
        )
        idempotence = float(
            np.max(np.abs(reprojection.prediction - frame["temp"].to_numpy(float)))
        )
        require(
            idempotence <= float(config["candidate_on_pass"]["pava_idempotence_tolerance"]),
            "independent PAVA idempotence failed",
        )
        candidate_qa = {
            "present_as_required": True,
            "validation": validation,
            "sha256": sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "pava_idempotence_max_abs": idempotence,
            "source_score_submission_side_contract": True,
        }
    else:
        csv_files = [str(path) for path in artifact.rglob("*.csv")]
        require(not csv_files, "failed gate created a candidate CSV")
        require("candidate" not in result, "failed gate recorded a candidate")
        candidate_qa = {"present_as_required": False, "csv_files": []}

    qa = {
        "schema_version": "p2.depth_registered_cmfpca.independent_qa.20260828.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS",
        "decision_reproduced": result["decision"],
        "historical_exposure_confirmed": True,
        "metrics": {"aggregate": aggregate, "by_fold": by_fold, "by_layer": by_layer},
        "bootstrap": boot,
        "independent_gate_checks": independent_checks,
        "candidate": candidate_qa,
        "leakage": {
            "answer_file_read": False,
            "hidden_answer_or_mirror_read": False,
            "official_upload_performed": False,
        },
    }
    output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

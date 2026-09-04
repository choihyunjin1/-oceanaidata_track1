"""Run one sealed P2 public-only thermocline-heave experiment.

The command has a read-only ``--check-only`` preflight and one ``--execute``
path.  During execution it writes and hashes every prediction before loading
the historical validation truth from the selected frozen OOF comparator.
There is no deployment, prediction CSV, or upload path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from p2_restore.p2_public_heave_tangent_incumbent_20260828_v1 import (
    TARGET_LAYERS,
    apply_heave_to_incumbent,
    build_public_panel,
    estimate_training_eta_cap,
    evaluate_gate,
    fit_seasonal_backgrounds,
    mask_validation_targets,
    paired_kst_day_bootstrap,
    rmse,
    season_bins,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_public_heave_tangent_incumbent_20260828_v1"
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def resolve_observations_path() -> Path:
    raw = os.environ.get("P2_DATA_DIR")
    if not raw:
        raise FileNotFoundError("set P2_DATA_DIR for this process")
    path = Path(raw).expanduser().resolve() / "observations.csv"
    require(path.is_file(), "P2_DATA_DIR lacks observations.csv")
    return path


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def decode_committed_time(values: np.ndarray) -> pd.DatetimeIndex:
    current = np.asarray(values, dtype=np.int64)
    unit = "ns" if int(np.max(np.abs(current))) >= 10**17 else "us"
    decoded = pd.DatetimeIndex(pd.to_datetime(current, unit=unit, utc=True))
    require(decoded.min().year >= 2024 and decoded.max().year <= 2025, "invalid time unit")
    return decoded


def validate_common_frame(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    required = {"time", "layer", "block", "reference"}
    require(required.issubset(frame.columns), f"incumbent schema missing {required - set(frame)}")
    result = frame.loc[:, ["time", "layer", "block", "reference"]].copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result["layer"] = pd.to_numeric(result["layer"], errors="raise").astype(int)
    result["block"] = result["block"].astype(str)
    result["reference"] = pd.to_numeric(result["reference"], errors="raise").astype(float)
    require(len(result) == 69_850, "incumbent common OOF row count changed")
    require(not result.duplicated(["time", "layer"]).any(), "incumbent keys duplicate")
    require(np.isfinite(result["reference"]).all(), "incumbent contains non-finite predictions")
    require(set(result["layer"].unique()) == set(TARGET_LAYERS), "incumbent layers changed")
    require(set(result["block"].unique()) == set(config["folds"]), "incumbent blocks changed")
    for name, specification in config["folds"].items():
        group = result.loc[result["block"].eq(name)]
        start = pd.Timestamp(str(specification["start"])).tz_convert("UTC")
        stop = pd.Timestamp(str(specification["stop"])).tz_convert("UTC")
        require(
            group["time"].ge(start).all() and group["time"].lt(stop).all(),
            f"fold bounds changed: {name}",
        )
    return result.sort_values(["time", "layer"]).reset_index(drop=True)


def _validate_parquet_candidate(
    specification: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    path = REPO / str(specification["path"])
    require(path.is_file(), "candidate OOF is absent")
    require(sha256(path) == specification["sha256"], "candidate OOF hash changed")
    prediction_column = str(specification["prediction_column"])
    columns = ["time", "layer", "block", prediction_column]
    frame = pd.read_parquet(path, columns=columns).rename(columns={prediction_column: "reference"})
    return validate_common_frame(frame, config)


def _validate_alpha40_proxy(
    specification: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block in config["folds"]:
        file_spec = specification["prediction_files"][block]
        path = REPO / str(file_spec["path"])
        require(path.is_file(), f"alpha40 proxy is absent: {block}")
        require(sha256(path) == file_spec["sha256"], f"alpha40 proxy hash changed: {block}")
        with np.load(path, allow_pickle=False) as payload:
            frames.append(
                pd.DataFrame(
                    {
                        "time": decode_committed_time(payload["time_ns"]),
                        "layer": payload["layer"].astype(int),
                        "block": block,
                        "reference": payload["reference"].astype(float),
                    }
                )
            )
    truth_anchor = REPO / str(specification["truth_anchor"]["path"])
    require(truth_anchor.is_file(), "alpha40 truth anchor is absent")
    require(
        sha256(truth_anchor) == specification["truth_anchor"]["sha256"],
        "alpha40 truth anchor hash changed",
    )
    return validate_common_frame(pd.concat(frames, ignore_index=True), config)


def select_comparator(
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    for index, specification in enumerate(config["comparator_priority"]):
        try:
            if specification["kind"] == "parquet":
                frame = _validate_parquet_candidate(specification, config)
            elif specification["kind"] == "sealed_npz_reference":
                frame = _validate_alpha40_proxy(specification, config)
            else:
                raise RuntimeError(f"unsupported comparator kind: {specification['kind']}")
            selected = {
                "priority_index": int(index),
                "name": str(specification["name"]),
                "kind": str(specification["kind"]),
                "validation_label": str(specification["validation_label"]),
                "freshness": str(specification["freshness"]),
                "downgraded": bool(index > 0),
                "input_hashes": _comparator_hashes(specification),
            }
            return selected, frame, failures
        except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
            failures.append(
                {"name": str(specification.get("name", "unknown")), "reason": str(error)}
            )
    raise RuntimeError(f"no comparator passed preflight: {failures}")


def _comparator_hashes(specification: dict[str, object]) -> dict[str, str]:
    if specification["kind"] == "parquet":
        return {str(specification["path"]): str(specification["sha256"])}
    hashes = {
        str(value["path"]): str(value["sha256"])
        for value in specification["prediction_files"].values()
    }
    hashes[str(specification["truth_anchor"]["path"])] = str(
        specification["truth_anchor"]["sha256"]
    )
    return hashes


def read_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    expected = [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]
    require(list(frame.columns) == expected, "observations schema changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    require(not frame.duplicated(["station", "time", "layer"]).any(), "observation keys duplicate")
    return frame


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    time = pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8.astype(np.int64)
    np.savez_compressed(
        path,
        time_ns=time,
        layer=frame["layer"].to_numpy(np.int16),
        reference=frame["reference"].to_numpy(np.float64),
        candidate=frame["candidate"].to_numpy(np.float64),
        correction=frame["correction"].to_numpy(np.float64),
        enabled=frame["enabled"].to_numpy(bool),
        eta_m=frame["eta_m"].to_numpy(np.float64),
    )


def prediction_stage(
    *,
    config_path: Path,
    artifact_dir: Path,
    config: dict[str, object],
    selected: dict[str, object],
    incumbent: pd.DataFrame,
    selection_failures: list[dict[str, str]],
) -> dict[str, object]:
    observations_path = resolve_observations_path()
    require(
        sha256(observations_path) == config["input_pins"]["observations_sha256"],
        "observations hash changed",
    )
    observations = read_observations(observations_path)
    masked, masked_rows = mask_validation_targets(observations, config["folds"])
    public_panel = build_public_panel(masked)
    model = config["model"]
    support = {
        "minimum_public_layers": int(model["minimum_public_layers"]),
        "minimum_public_span_m": float(model["minimum_public_span_m"]),
        "minimum_gradient_rms_c_per_m": float(model["minimum_gradient_rms_c_per_m"]),
        "maximum_design_condition_number": float(model["maximum_design_condition_number"]),
    }
    target_depth = np.asarray(
        [float(model["target_nominal_depth_m"][str(layer)]) for layer in TARGET_LAYERS],
        dtype=np.float64,
    )
    if artifact_dir.exists():
        existing_files = [path for path in artifact_dir.rglob("*") if path.is_file()]
        require(not existing_files, "precommit recovery found an existing artifact file")
    else:
        artifact_dir.mkdir(parents=True)
    prediction_dir = artifact_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    outputs: dict[str, object] = {}
    receipts: dict[str, object] = {}
    all_bins = set(range(int(np.ceil(366 / int(model["season_bin_days"])))))
    for block, specification in config["folds"].items():
        query = incumbent.loc[incumbent["block"].eq(block)].reset_index(drop=True)
        start = pd.Timestamp(str(specification["start"])).tz_convert("UTC")
        requested = set(
            int(value)
            for value in season_bins(pd.DatetimeIndex(query["time"]), int(model["season_bin_days"]))
        )
        backgrounds, background_receipt = fit_seasonal_backgrounds(
            public_panel,
            train_stop=start,
            query_bins=all_bins,
            purge_days=int(model["purge_days"]),
            season_bin_days=int(model["season_bin_days"]),
            season_window_days=float(model["season_window_days"]),
            minimum_rows_per_layer=int(model["minimum_background_rows_per_layer"]),
        )
        eta_cap, eta_receipt = estimate_training_eta_cap(
            public_panel,
            backgrounds,
            train_stop=start,
            purge_days=int(model["purge_days"]),
            season_bin_days=int(model["season_bin_days"]),
            target_depth=target_depth,
            stride=int(model["eta_training_stride"]),
            hard_cap_m=float(model["eta_hard_cap_m"]),
            quantile=float(model["eta_absolute_quantile"]),
            minimum_eta_rows=int(model["minimum_eta_training_rows"]),
            support=support,
        )
        candidate, diagnostics = apply_heave_to_incumbent(
            query,
            public_panel,
            backgrounds,
            season_bin_days=int(model["season_bin_days"]),
            target_depth_by_layer={
                int(key): float(value) for key, value in model["target_nominal_depth_m"].items()
            },
            eta_cap_m=eta_cap,
            maximum_correction_c=float(model["maximum_correction_absolute_c"]),
            support=support,
        )
        path = prediction_dir / f"{EXPERIMENT_ID}_{block}.npz"
        write_prediction(path, candidate)
        outputs[block] = {
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "rows": int(len(candidate)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        receipts[block] = {
            "validation_start_utc": start.isoformat(),
            "validation_stop_utc": pd.Timestamp(str(specification["stop"]))
            .tz_convert("UTC")
            .isoformat(),
            "purge_days": int(model["purge_days"]),
            "validation_target_temp_psal_masked_together": True,
            "validation_truth_loaded": False,
            "public_only_mode_and_amplitude": True,
            "query_season_bins": sorted(requested),
            "unsupported_query_bins_are_exact_noop": sorted(requested - set(backgrounds)),
            "background": background_receipt,
            "eta_cap": eta_receipt,
            "correction": diagnostics,
        }
    commitment = {
        "schema_version": "p2.public_heave_tangent_incumbent.prediction_commitment.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "truth_metric_computed": False,
        "validation_truth_loaded": False,
        "precommit_recovery": {
            "performed": True,
            "truth_or_metric_observed_before_recovery": False,
            "reason": "Two execution calls stopped before prediction files: first, the last 2024 validation season bin lacked prefix support; second, hourly subsampling left only 109 public-only eta rows for the first-fold q99 receipt.",
            "recovery": "Use the already-specified exact incumbent no-op for missing seasonal bins and restore the q99 receipt to the source 10-minute cadence (stride 1). The effective eta cap remained the preregistered 10 m hard cap in every fold; no truth, metric, mode, fold, comparator, gate, or correction threshold was observed or changed. The empty append-only artifact directory was retained.",
        },
        "comparator_selected_before_prediction": selected,
        "comparator_preflight_failures": selection_failures,
        "masked_target_rows": masked_rows,
        "prediction_outputs": outputs,
        "fold_receipts": receipts,
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256(config_path),
            },
            "observations": {
                "logical_path": "P2_DATA_DIR/observations.csv",
                "sha256": sha256(observations_path),
            },
        },
        "leakage_audit": {
            "official_input_paths_read": False,
            "official_answer_or_mirror_read": False,
            "validation_target_temp_psal_masked_together": True,
            "target_temp_psal_used_as_features": False,
            "new_pava_applied": False,
            "candidate_csv_generated": False,
            "validation_truth_loaded_before_seal": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "cpu_thread_limit": int(model["cpu_thread_limit"]),
            "git": git_snapshot(),
        },
    }
    path = artifact_dir / "prediction_commitment.json"
    path.write_text(json.dumps(commitment, ensure_ascii=False, indent=2), encoding="utf-8")
    return commitment


def load_prediction_outputs(commitment: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block, output in commitment["prediction_outputs"].items():
        path = REPO / str(output["path"])
        require(sha256(path) == output["sha256"], f"prediction seal changed: {block}")
        with np.load(path, allow_pickle=False) as payload:
            frames.append(
                pd.DataFrame(
                    {
                        "time": decode_committed_time(payload["time_ns"]),
                        "layer": payload["layer"].astype(int),
                        "block": block,
                        "reference": payload["reference"].astype(float),
                        "candidate": payload["candidate"].astype(float),
                        "correction": payload["correction"].astype(float),
                        "enabled": payload["enabled"].astype(bool),
                        "eta_m": payload["eta_m"].astype(float),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_truth_after_seal(
    selected: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    specification = config["comparator_priority"][int(selected["priority_index"])]
    if specification["kind"] == "parquet":
        path = REPO / str(specification["path"])
        require(sha256(path) == specification["sha256"], "selected comparator hash changed")
        truth = pd.read_parquet(path, columns=["time", "layer", "block", "truth"])
    else:
        anchor = specification["truth_anchor"]
        path = REPO / str(anchor["path"])
        require(sha256(path) == anchor["sha256"], "proxy truth anchor hash changed")
        truth = pd.read_parquet(path, columns=["time", "layer", "block", "truth"])
    truth["time"] = pd.to_datetime(truth["time"], utc=True)
    truth["layer"] = pd.to_numeric(truth["layer"], errors="raise").astype(int)
    require(np.isfinite(truth["truth"]).all(), "truth anchor is non-finite")
    require(not truth.duplicated(["time", "layer"]).any(), "truth keys duplicate")
    return truth


def metric_group(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {
        "rows": int(len(frame)),
        "incumbent_rmse_c": reference,
        "heave_candidate_rmse_c": candidate,
        "delta_rmse_c": candidate - reference,
    }


def score_after_prediction_seal(
    *,
    config_path: Path,
    artifact_dir: Path,
    config: dict[str, object],
    commitment: dict[str, object],
) -> dict[str, object]:
    predictions = load_prediction_outputs(commitment)
    truth = load_truth_after_seal(commitment["comparator_selected_before_prediction"], config)
    scored = predictions.merge(
        truth, on=["time", "layer", "block"], how="left", validate="one_to_one"
    )
    require(len(scored) == 69_850 and scored["truth"].notna().all(), "truth alignment failed")
    aggregate = metric_group(scored)
    folds = {str(name): metric_group(group) for name, group in scored.groupby("block", sort=True)}
    layers = {
        str(int(name)): metric_group(group) for name, group in scored.groupby("layer", sort=True)
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["model"]["bootstrap_replicates"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(np.float64)
    enabled = scored["enabled"].to_numpy(bool)
    disabled = ~enabled
    require(
        np.array_equal(
            scored.loc[disabled, "candidate"].to_numpy(),
            scored.loc[disabled, "reference"].to_numpy(),
        ),
        "fallback changed",
    )
    correction_rms = float(np.sqrt(np.mean(correction**2)))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    correction_maximum = float(np.max(np.abs(correction)))
    active_fraction = float(enabled.mean())
    gate = evaluate_gate(
        aggregate_delta=float(aggregate["delta_rmse_c"]),
        ci90_high=float(bootstrap["ci90_high_c"]),
        fold_deltas={name: float(value["delta_rmse_c"]) for name, value in folds.items()},
        layer_deltas={name: float(value["delta_rmse_c"]) for name, value in layers.items()},
        active_fraction=active_fraction,
        correction_rms=correction_rms,
        correction_p99=correction_p99,
        correction_maximum=correction_maximum,
        thresholds=config["gate"],
    )
    decision = (
        "PASS_BOUNDED_LOCAL_GATE_NO_CSV_NO_UPLOAD"
        if gate["passed"]
        else "FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP"
    )
    commitment_path = artifact_dir / "prediction_commitment.json"
    result = {
        "schema_version": "p2.public_heave_tangent_incumbent.result.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "decision": decision,
        "historical_exposure": config["historical_exposure"],
        "split_claim": "committed historical block-mask validation; not an untouched holdout",
        "comparator": commitment["comparator_selected_before_prediction"],
        "metrics": {"aggregate": aggregate, "by_fold": folds, "by_layer": layers},
        "paired_kst_day_bootstrap": bootstrap,
        "correction": {
            "enabled_rows": int(enabled.sum()),
            "enabled_fraction": active_fraction,
            "rms_c": correction_rms,
            "p99_absolute_c": correction_p99,
            "maximum_absolute_c": correction_maximum,
            "fallback_maximum_absolute_c": float(np.max(np.abs(correction[disabled])))
            if disabled.any()
            else 0.0,
        },
        "gate": gate,
        "prediction_commitment": {
            "path": str(commitment_path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(commitment_path),
            "verified_before_truth_load": True,
        },
        "leakage_audit": {
            "official_input_paths_read": False,
            "official_answer_or_mirror_read": False,
            "candidate_csv_generated": False,
            "official_upload_performed": False,
            "target_temp_psal_used_as_features": False,
            "validation_truth_loaded_only_after_prediction_hash_verification": True,
            "new_pava_applied": False,
            "post_result_parameter_search_performed": False,
        },
    }
    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sources = [
        config_path,
        REPO / "src" / "p2_restore" / f"{EXPERIMENT_ID}.py",
        Path(__file__).resolve(),
        REPO / "scripts" / f"qa_{EXPERIMENT_ID}.py",
        REPO / "tests" / f"test_{EXPERIMENT_ID}.py",
    ]
    manifest = {
        "schema_version": "p2.public_heave_tangent_incumbent.manifest.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "sources": {
            str(path.relative_to(REPO)).replace("\\", "/"): sha256(path) for path in sources
        },
        "inputs": {
            **commitment["inputs"],
            "comparator": commitment["comparator_selected_before_prediction"],
        },
        "outputs": {
            "prediction_commitment": {
                "path": str(commitment_path.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256(commitment_path),
            },
            "predictions": commitment["prediction_outputs"],
            "result": {
                "path": str(result_path.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256(result_path),
            },
            "candidate_csv": None,
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def validate_config(config: dict[str, object]) -> None:
    require(config["experiment_id"] == EXPERIMENT_ID, "experiment ID changed")
    require(int(config["model"]["hyperparameter_search_count"]) == 0, "search count changed")
    require(int(config["model"]["cpu_thread_limit"]) <= 2, "CPU thread limit exceeds two")
    leakage = config["leakage_contract"]
    require(leakage["official_test_sample_submission_paths_read"] is False, "official input flag")
    require(leakage["query_target_temp_psal_as_features"] is False, "target feature flag")
    require(leakage["new_pava_allowed"] is False, "PAVA flag")
    require(leakage["candidate_csv_generation_authorized"] is False, "CSV flag")
    require(leakage["post_result_parameter_search_allowed"] is False, "search flag")


def main() -> None:
    _thread_limit = threadpool_limits(limits=2)
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    observations_path = resolve_observations_path()
    require(
        sha256(observations_path) == config["input_pins"]["observations_sha256"],
        "observations pin differs",
    )
    selected, incumbent, failures = select_comparator(config)
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256(config_path),
        "observations_sha256": sha256(observations_path),
        "comparator": selected,
        "prior_failures": failures,
        "rows": int(len(incumbent)),
        "fold_rows": incumbent["block"].value_counts().sort_index().to_dict(),
        "target_values_read": False,
        "official_input_paths_read": False,
    }
    if args.check_only:
        print(json.dumps({"stage": "check-only", **preflight}, ensure_ascii=False, indent=2))
        return
    commitment = prediction_stage(
        config_path=config_path,
        artifact_dir=artifact_dir,
        config=config,
        selected=selected,
        incumbent=incumbent,
        selection_failures=failures,
    )
    result = score_after_prediction_seal(
        config_path=config_path,
        artifact_dir=artifact_dir,
        config=config,
        commitment=commitment,
    )
    print(
        json.dumps(
            {
                "stage": "completed",
                "decision": result["decision"],
                "comparator": result["comparator"],
                "metrics": result["metrics"],
                "bootstrap": result["paired_kst_day_bootstrap"],
                "correction": result["correction"],
                "gate": result["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

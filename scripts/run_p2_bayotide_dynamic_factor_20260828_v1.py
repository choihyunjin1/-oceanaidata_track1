"""Run the authorized local-only P2 fixed-factor BayOTIDE-style pilot once."""

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
from threadpoolctl import threadpool_limits

from p2_restore.p2_bayotide_dynamic_factor_20260828_v1 import (
    TARGET_LAYERS,
    build_registered_panel,
    evaluate_gate,
    fit_fixed_dynamic_factor,
    fold_masks,
    guarded_temperature_candidate,
    paired_kst_day_bootstrap,
    rmse,
    smooth_dynamic_factor,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_bayotide_dynamic_factor_20260828_v1"
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    return parser.parse_args()


def resolve_observations_path() -> Path:
    raw = os.environ.get("P2_DATA_DIR")
    require(bool(raw), "set P2_DATA_DIR")
    root = Path(str(raw)).expanduser().resolve()
    path = root / "observations.csv"
    require(path.is_file(), "P2_DATA_DIR/observations.csv is absent")
    return path


def read_observations(path: Path) -> pd.DataFrame:
    columns = ["year", "time", "layer", "temp", "psal", "depth", "nominal_depth"]
    frame = pd.read_csv(path, usecols=columns)
    require(len(frame) == 789_408, "observation row count changed")
    require(not frame.duplicated(["year", "time", "layer"]).any(), "observation keys duplicate")
    return frame


def validate_config(config: dict[str, object]) -> None:
    require(config["experiment_id"] == EXPERIMENT_ID, "experiment id changed")
    require(config["status"] == "AUTHORIZED_LOCAL_ONE_SHOT_20260828", "local authorization absent")
    require(config["authorization"]["official_input_or_upload_authorized"] is False, "scope changed")
    model = config["model"]
    require(int(model["trend_factors"]) == 3, "trend factor count changed")
    require(model["trend_kernel"] == "Matern_3_2", "trend kernel changed")
    require(tuple(float(value) for value in model["trend_lengthscale_hours"]) == (6.0, 48.0, 336.0), "trend lengthscales changed")
    periodic = tuple(float(value["period_hours"]) for value in model["periodic_factors"])
    require(periodic == (12.42, 24.0), "periodic factors changed")
    require(int(model["factor_or_kernel_search_count"]) == 0, "search is forbidden")
    require(model["inference"] == "forward_filter_plus_RTS_smoother", "inference changed")
    require(model["uncertainty_fallback"] == "exact_incumbent_no_op", "fallback changed")
    require(config["validation"]["purge_days"] == 7, "purge changed")
    require(config["validation"]["comparator"] == "p2_extrapolated_soft_gate_v2", "comparator changed")
    policy = config["execution_policy"]
    require(policy["single_bounded_run"] is True, "single-run lock changed")
    require(policy["result_based_factor_kernel_gate_rerun"] is False, "rerun policy changed")
    require(policy["official_upload_authorized"] is False, "upload is forbidden")


def comparator_preflight(config: dict[str, object], *, include_truth: bool) -> pd.DataFrame:
    pin = config["input_pins"]
    path = REPO / str(pin["comparator_path"])
    require(path.is_file() and sha256(path) == pin["comparator_sha256"], "comparator pin changed")
    columns = ["time", "layer", "block", "prediction"]
    if include_truth:
        columns.append("truth")
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.rename(columns={"prediction": "reference"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    require(len(frame) == int(config["validation"]["expected_oof_rows"]), "OOF rows changed")
    require(not frame.duplicated(["time", "layer"]).any(), "OOF keys duplicate")
    require(frame["layer"].isin(TARGET_LAYERS).all(), "OOF layer changed")
    numeric = ["reference"] + (["truth"] if include_truth else [])
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), "OOF values non-finite")
    return frame.sort_values(["time", "layer"]).reset_index(drop=True)


def git_snapshot() -> dict[str, object]:
    def command(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "branch": command("branch", "--show-current"),
        "head": command("rev-parse", "HEAD"),
        "dirty": bool(command("status", "--short")),
    }


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8,
        layer=frame["layer"].to_numpy(np.int16),
        reference=frame["reference"].to_numpy(np.float64),
        dynamic=frame["dynamic"].to_numpy(np.float64),
        candidate=frame["candidate"].to_numpy(np.float64),
        posterior_sd_c=frame["posterior_sd_c"].to_numpy(np.float64),
        active=frame["active"].to_numpy(bool),
    )


def prediction_stage(
    *,
    config_path: Path,
    artifact_dir: Path,
    observations_path: Path,
    observations: pd.DataFrame,
    comparator: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    recovery = False
    if artifact_dir.exists():
        existing = tuple(path.relative_to(artifact_dir).as_posix() for path in artifact_dir.rglob("*"))
        require(
            existing in ((), ("predictions",)),
            "single bounded execution artifact already contains a sealed output",
        )
        recovery = True
    else:
        artifact_dir.mkdir(parents=True)
    prediction_dir = artifact_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    fit = config["model"]["fixed_fit"]
    periods = tuple(float(value["period_hours"]) for value in config["model"]["periodic_factors"])
    lengthscales = tuple(float(value) for value in config["model"]["trend_lengthscale_hours"])
    outputs: dict[str, object] = {}
    receipts: dict[str, object] = {}
    for block, interval in config["validation"]["folds"].items():
        fold = comparator.loc[comparator["block"].eq(block)].copy()
        require(not fold.empty, f"comparator block absent: {block}")
        year = int(str(interval[0])[:4])
        panel = build_registered_panel(observations, year)
        training, validation, purged = fold_masks(
            panel.times, interval[0], interval[1], purge_days=int(config["validation"]["purge_days"])
        )
        model, registered = fit_fixed_dynamic_factor(
            panel,
            training,
            trend_lengthscales_hours=lengthscales,
            periods_hours=periods,
            completion_iterations=int(fit["matrix_completion_iterations"]),
            minimum_channel_coverage=float(fit["minimum_channel_coverage"]),
            observation_noise_floor=float(fit["observation_noise_floor_normalized"]),
            periodic_damping=float(fit["periodic_damping_per_10min_step"]),
            posterior_multiplier=float(fit["posterior_sd_training_residual_multiplier"]),
            posterior_absolute_cap_c=float(fit["posterior_sd_absolute_cap_c"]),
        )
        predicted, posterior, observed = smooth_dynamic_factor(panel, registered, model, purged)
        row_index = panel.times.get_indexer(pd.DatetimeIndex(fold["time"]))
        require(np.all(row_index >= 0), f"{block} OOF time missing from panel")
        require(validation[row_index].all(), f"{block} OOF time escaped validation")
        temp_channels = np.asarray(
            [
                model.channel_variables.index("temp", 0)
                + list(model.channel_layers[: len(panel.layers)]).index(layer)
                for layer in TARGET_LAYERS
            ],
            dtype=int,
        )
        # channel_variables is variable-major and the first support block is temperature.
        temp_channels = np.asarray(
            [
                index
                for layer in TARGET_LAYERS
                for index, (variable, current_layer) in enumerate(
                    zip(model.channel_variables, model.channel_layers, strict=True)
                )
                if variable == "temp" and current_layer == layer
            ],
            dtype=int,
        )
        require(len(temp_channels) == 3, "target temperature channel mapping failed")
        unique_times = pd.DatetimeIndex(fold["time"].drop_duplicates()).sort_values()
        unique_rows = panel.times.get_indexer(unique_times)
        base = fold.pivot(index="time", columns="layer", values="reference").reindex(
            index=unique_times, columns=TARGET_LAYERS
        )
        base_values = base.to_numpy(float)
        public_channels = np.asarray(
            [layer not in TARGET_LAYERS for layer in model.channel_layers], dtype=bool
        )
        public_count = observed[unique_rows][:, public_channels].sum(axis=1)
        guarded = guarded_temperature_candidate(
            incumbent=base_values,
            dynamic_temperature=predicted[unique_rows][:, temp_channels],
            posterior_sd_c=posterior[unique_rows][:, temp_channels],
            public_observed_counts=public_count,
            posterior_guard_c=model.posterior_guard_c[temp_channels],
            minimum_public_channels=int(fit["minimum_observed_public_channels"]),
        )
        candidate_wide = pd.DataFrame(guarded.candidate, index=unique_times, columns=TARGET_LAYERS)
        dynamic_wide = pd.DataFrame(guarded.dynamic, index=unique_times, columns=TARGET_LAYERS)
        posterior_wide = pd.DataFrame(guarded.posterior_sd_c, index=unique_times, columns=TARGET_LAYERS)
        active_wide = pd.DataFrame(guarded.active, index=unique_times, columns=TARGET_LAYERS)
        keyed = fold.set_index(["time", "layer"])
        keyed["candidate"] = [candidate_wide.at[time, layer] for time, layer in keyed.index]
        keyed["dynamic"] = [dynamic_wide.at[time, layer] for time, layer in keyed.index]
        keyed["posterior_sd_c"] = [posterior_wide.at[time, layer] for time, layer in keyed.index]
        keyed["active"] = [active_wide.at[time, layer] for time, layer in keyed.index]
        candidate = keyed.reset_index()
        require(np.isfinite(candidate[["candidate", "dynamic", "posterior_sd_c"]]).all().all(), f"{block} prediction non-finite")
        require(
            np.array_equal(
                candidate.loc[~candidate["active"], "candidate"].to_numpy(),
                candidate.loc[~candidate["active"], "reference"].to_numpy(),
            ),
            f"{block} fallback differs",
        )
        path = prediction_dir / f"{block}.npz"
        write_prediction(path, candidate)
        outputs[block] = {
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "rows": int(len(candidate)),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        target_channel_mask = np.asarray(
            [layer in TARGET_LAYERS for layer in model.channel_layers], dtype=bool
        )
        receipts[block] = {
            "year_regime": year,
            "training_rows": int(training.sum()),
            "validation_rows": int(validation.sum()),
            "purged_rows": int(purged.sum()),
            "registered_supports": int(len(panel.layers)),
            "joint_temperature_salinity_channels": int(len(model.channel_layers)),
            "target_update_cells_masked": int(purged.sum() * target_channel_mask.sum()),
            "trend_factors": 3,
            "periodic_factors_hours": list(periods),
            "state_dimension": int(model.transition.shape[0]),
            "active_rows": int(candidate["active"].sum()),
            "active_fraction": float(candidate["active"].mean()),
            "posterior_sd_guard_applied": True,
            "masked_target_values_used_in_fit_or_update": False,
        }
    commitment = {
        "schema_version": "p2.bayotide_dynamic_factor.prediction_commitment.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "precommit_recovery": {
            "performed": recovery,
            "reason": "NaN-aware exact-fallback assertion repair before any prediction seal or truth metric",
            "truth_metric_observed_before_recovery": False,
            "model_or_gate_changed": False,
        },
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "prediction_outputs": outputs,
        "fold_receipts": receipts,
        "inputs": {
            "config": {"path": str(config_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256(config_path)},
            "observations": {"logical_path": "P2_DATA_DIR/observations.csv", "sha256": sha256(observations_path)},
            "comparator": {"path": config["input_pins"]["comparator_path"], "sha256": config["input_pins"]["comparator_sha256"]},
        },
        "leakage_audit": {
            "official_test_sample_submission_paths_read": False,
            "candidate_csv_generated": False,
            "validation_target_temp_psal_masked_together": True,
            "purged_target_temp_psal_measurement_updates": False,
            "factor_or_kernel_search_count": 0,
            "truth_metric_computed_before_prediction_hash": False,
        },
        "runtime": {"python": platform.python_version(), "cpu_thread_limit": int(fit["cpu_thread_limit"]), "git": git_snapshot()},
    }
    path = artifact_dir / "prediction_commitment.json"
    path.write_text(json.dumps(commitment, ensure_ascii=False, indent=2), encoding="utf-8")
    return commitment


def load_predictions(commitment: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block, output in commitment["prediction_outputs"].items():
        path = REPO / output["path"]
        require(sha256(path) == output["sha256"], f"prediction hash changed: {block}")
        with np.load(path, allow_pickle=False) as payload:
            frames.append(
                pd.DataFrame(
                    {
                        "time": pd.to_datetime(payload["time_ns"], utc=True),
                        "layer": payload["layer"].astype(int),
                        "block": block,
                        "reference": payload["reference"].astype(float),
                        "dynamic": payload["dynamic"].astype(float),
                        "candidate": payload["candidate"].astype(float),
                        "posterior_sd_c": payload["posterior_sd_c"].astype(float),
                        "active": payload["active"].astype(bool),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def metric(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {"rows": int(len(frame)), "incumbent_rmse_c": reference, "candidate_rmse_c": candidate, "delta_rmse_c": candidate - reference}


def score_after_seal(
    *, config_path: Path, artifact_dir: Path, config: dict[str, object], commitment: dict[str, object]
) -> dict[str, object]:
    predictions = load_predictions(commitment)
    truth = comparator_preflight(config, include_truth=True)[["time", "layer", "block", "truth"]]
    scored = predictions.merge(truth, on=["time", "layer", "block"], how="left", validate="one_to_one")
    require(len(scored) == int(config["validation"]["expected_oof_rows"]), "scored row count changed")
    require(scored["truth"].notna().all(), "truth alignment failed")
    aggregate = metric(scored)
    folds = {str(name): metric(group) for name, group in scored.groupby("block", sort=True)}
    layers = {str(int(name)): metric(group) for name, group in scored.groupby("layer", sort=True)}
    fit = config["model"]["fixed_fit"]
    bootstrap = paired_kst_day_bootstrap(scored, replicates=int(fit["bootstrap_replicates"]), seed=int(fit["bootstrap_seed"]))
    gate = evaluate_gate(
        aggregate_delta=float(aggregate["delta_rmse_c"]),
        ci90_high=float(bootstrap["ci90_high_c"]),
        fold_deltas={name: float(value["delta_rmse_c"]) for name, value in folds.items()},
        layer_deltas={name: float(value["delta_rmse_c"]) for name, value in layers.items()},
        thresholds=config["promotion_gate"],
        posterior_guard_applied=True,
    )
    active = scored["active"].to_numpy(bool)
    require(np.array_equal(scored.loc[~active, "candidate"].to_numpy(), scored.loc[~active, "reference"].to_numpy()), "fallback changed after seal")
    result = {
        "schema_version": "p2.bayotide_dynamic_factor.result.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "decision": "PASS_LOCAL_GATE_NO_CSV_NO_UPLOAD" if gate["passed"] else "FAIL_GATE_STOP_NO_CSV_NO_RERUN",
        "split_claim": "historical exposed three-block OOF; not an untouched holdout",
        "metrics": {"aggregate": aggregate, "by_fold": folds, "by_layer": layers},
        "paired_kst_day_bootstrap": bootstrap,
        "activity": {
            "rows": int(len(scored)),
            "active_rows": int(active.sum()),
            "active_fraction": float(active.mean()),
            "posterior_sd_median_c": float(np.median(scored["posterior_sd_c"])),
            "posterior_sd_p95_c": float(np.quantile(scored["posterior_sd_c"], 0.95)),
            "replacement_rms_c": float(np.sqrt(np.mean(np.square(scored["candidate"] - scored["reference"])))),
        },
        "gate": gate,
        "prediction_commitment": {"path": str((artifact_dir / "prediction_commitment.json").relative_to(REPO)).replace("\\", "/"), "sha256": sha256(artifact_dir / "prediction_commitment.json"), "verified_before_truth_load": True},
        "leakage_audit": {
            "official_test_sample_submission_paths_read": False,
            "candidate_csv_generated": False,
            "official_upload_performed": False,
            "factor_or_kernel_search_count": 0,
            "result_based_rerun_performed": False,
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
        "schema_version": "p2.bayotide_dynamic_factor.manifest.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": result["decision"],
        "sources": {str(path.relative_to(REPO)).replace("\\", "/"): sha256(path) for path in sources},
        "inputs": commitment["inputs"],
        "outputs": {
            "prediction_commitment": {"path": result["prediction_commitment"]["path"], "sha256": result["prediction_commitment"]["sha256"]},
            "predictions": commitment["prediction_outputs"],
            "result": {"path": str(result_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256(result_path)},
            "candidate_csv": None,
        },
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    observations_path = resolve_observations_path()
    require(sha256(observations_path) == config["input_pins"]["observations_sha256"], "observation pin changed")
    comparator = comparator_preflight(config, include_truth=False)
    preflight = {
        "stage": "check-only",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256(config_path),
        "observations_sha256": sha256(observations_path),
        "comparator_sha256": config["input_pins"]["comparator_sha256"],
        "rows": int(len(comparator)),
        "fold_rows": comparator["block"].value_counts().sort_index().to_dict(),
        "factor_or_kernel_search_count": 0,
        "official_input_paths_read": False,
        "candidate_csv_generated": False,
    }
    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    observations = read_observations(observations_path)
    with threadpool_limits(limits=int(config["model"]["fixed_fit"]["cpu_thread_limit"])):
        commitment = prediction_stage(
            config_path=config_path,
            artifact_dir=artifact_dir,
            observations_path=observations_path,
            observations=observations,
            comparator=comparator,
            config=config,
        )
        result = score_after_seal(config_path=config_path, artifact_dir=artifact_dir, config=config, commitment=commitment)
    print(json.dumps({"stage": "completed", "decision": result["decision"], "metrics": result["metrics"], "bootstrap": result["paired_kst_day_bootstrap"], "activity": result["activity"], "gate": result["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

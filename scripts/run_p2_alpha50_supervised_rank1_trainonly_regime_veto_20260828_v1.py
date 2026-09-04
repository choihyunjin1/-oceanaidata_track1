"""Run the sealed P2 train-only conditional-benefit regime veto pilot."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for directory in (ROOT, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel  # noqa: E402
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    bounded_profile_correction,
    paired_kst_day_bootstrap,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.supervised_rank1_functional_residual import (  # noqa: E402
    TARGET_LAYERS,
    SupervisedRank1Residual,
    build_public_functional_features,
)
from p2_restore.trainonly_regime_veto import (  # noqa: E402
    season_bin,
    trainonly_regime_decisions,
)
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)

EXPERIMENT_ID = "p2_alpha50_supervised_rank1_trainonly_regime_veto_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


def load_config(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise base.ContractError("experiment ID drifted")
    policy = config["execution_policy"]
    if any(
        (
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
            policy["result_based_retry"],
        )
    ):
        raise base.ContractError("forbidden official access or retry was enabled")
    records: list[dict[str, Any]] = []
    base_experiment = config["base_experiment"]
    records.extend(base_experiment[key] for key in ("config", "runner", "result", "commitment"))
    records.extend(base_experiment["predictions"].values())
    for record in records:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise base.ContractError(f"immutable input changed: {path}")
    base_config = json.loads((ROOT / base_experiment["config"]["path"]).read_text(encoding="utf-8"))
    observations_path = data_dir / base_config["source_observations"]["filename"]
    if (
        not observations_path.is_file()
        or base.sha256_file(observations_path) != base_config["source_observations"]["sha256"]
    ):
        raise base.ContractError("observations.csv pin changed")
    return config, base_config


def _profile_prediction(
    *,
    query: pd.DataFrame,
    query_features: pd.DataFrame,
    fitted: SupervisedRank1Residual,
    reference: np.ndarray,
    endpoints: pd.DataFrame,
    model_config: dict[str, Any],
) -> pd.DataFrame:
    query_times = pd.DatetimeIndex(sorted(query["time"].unique()))
    profile_prediction, profile_enabled, leverage = fitted.predict(query_features, query_times)
    layer_to_column = {layer: index for index, layer in enumerate(TARGET_LAYERS)}
    time_to_row = {timestamp: index for index, timestamp in enumerate(query_times)}
    raw_correction = np.zeros(len(query), dtype=np.float64)
    enabled = np.zeros(len(query), dtype=bool)
    row_leverage = np.full(len(query), np.inf, dtype=np.float64)
    for row, values in enumerate(query.itertuples(index=False)):
        profile_row = time_to_row[pd.Timestamp(values.time)]
        column = layer_to_column[int(values.layer)]
        raw_correction[row] = profile_prediction[profile_row, column]
        enabled[row] = bool(profile_enabled[profile_row])
        row_leverage[row] = leverage[profile_row]
    correction, _ = bounded_profile_correction(
        raw_correction,
        enabled,
        rms_cap=float(model_config["correction_rms_cap_c"]),
        p99_cap=float(model_config["correction_absolute_cap_c"]),
    )
    projected = project_profiles_vectorized(query, reference + correction, endpoints).prediction
    actual_correction = projected - reference
    if np.max(np.abs(actual_correction[~enabled]), initial=0.0) > 1e-12:
        raise base.ContractError("unsupported inner profile changed after projection")
    result = query[["time", "layer"]].copy()
    result["reference"] = reference
    result["candidate"] = projected
    result["correction"] = actual_correction
    result["enabled"] = enabled
    result["leverage"] = row_leverage
    return result


def inner_oof_block(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    held_block: str,
    base_config: dict[str, Any],
    observations: pd.DataFrame,
    functional_features: pd.DataFrame,
    anchor_path: Path,
) -> pd.DataFrame:
    """Predict one outer-training block while excluding its labels from every fit."""

    outer_start, outer_stop = base.utc(outer_spec["start"]), base.utc(outer_spec["stop"])
    held_start, held_stop = map(base.utc, base_config["block_bounds"][held_block])
    fit_blocks = [name for name in outer_spec["training_blocks"] if name != held_block]
    if not fit_blocks:
        raise base.ContractError(f"{outer_name}/{held_block} has no inner fit blocks")
    masked = observations.copy()
    target = masked["layer"].isin(TARGET_LAYERS)
    outer_mask = masked["time"].ge(outer_start) & masked["time"].lt(outer_stop) & target
    held_mask = masked["time"].ge(held_start) & masked["time"].lt(held_stop) & target
    masked.loc[outer_mask | held_mask, ["temp", "psal"]] = np.nan
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    query = base.add_metadata(base.block_anchor(anchor_path, held_block, include_truth=False), observations)
    reference, _ = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=outer_start,
        config=base_config,
        exclude=(held_start, held_stop),
    )
    training_parts: list[pd.DataFrame] = []
    for training_block in fit_blocks:
        training = base.add_metadata(
            base.block_anchor(anchor_path, training_block, include_truth=True), observations
        )
        if not training["time"].lt(outer_start).all():
            raise base.ContractError("inner training labels cross outer boundary")
        bounds = base_config["block_bounds"][training_block]
        inner_reference, _ = base.alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=outer_start,
            config=base_config,
            exclude=(base.utc(bounds[0]), base.utc(bounds[1])),
        )
        training["residual"] = training["truth"].to_numpy(dtype=np.float64) - inner_reference
        training_parts.append(training)
    training = pd.concat(training_parts, ignore_index=True)
    train_times, response = base.profile_response(training)
    train_features = base.align_features(functional_features, train_times)
    valid = train_features["public_profile_valid"].to_numpy(dtype=bool)
    fitted = SupervisedRank1Residual.fit(
        train_features.loc[valid].reset_index(drop=True), response[valid], train_times[valid]
    )
    query_times = pd.DatetimeIndex(sorted(query["time"].unique()))
    prediction = _profile_prediction(
        query=query,
        query_features=base.align_features(functional_features, query_times),
        fitted=fitted,
        reference=reference,
        endpoints=endpoints,
        model_config=base_config["model"],
    )
    truth = base.block_anchor(anchor_path, held_block, include_truth=True)[["time", "layer", "truth"]]
    prediction = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
    if prediction["truth"].isna().any():
        raise base.ContractError("inner OOF truth binding failed")
    prediction["source_block"] = held_block
    prediction["outer_fold"] = outer_name
    return prediction


def read_frozen_prediction(record: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / record["path"]
    if base.sha256_file(path) != record["sha256"]:
        raise base.ContractError("frozen base prediction changed")
    with np.load(path, allow_pickle=False) as payload:
        return pd.DataFrame(
            {
                "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                "layer": payload["layer"].astype(int),
                "reference": payload["reference"],
                "original_candidate": payload["candidate"],
                "original_correction": payload["correction"],
                "original_enabled": payload["enabled"].astype(bool),
            }
        )


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8,
        layer=frame["layer"].to_numpy(dtype=np.int16),
        reference=frame["reference"].to_numpy(dtype=np.float64),
        candidate=frame["candidate"].to_numpy(dtype=np.float64),
        correction=frame["correction"].to_numpy(dtype=np.float64),
        original_enabled=frame["original_enabled"].to_numpy(dtype=bool),
        regime_enabled=frame["regime_enabled"].to_numpy(dtype=bool),
        final_enabled=frame["final_enabled"].to_numpy(dtype=bool),
    )


def run(config: dict[str, Any], base_config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    predictions_directory = output_directory / "predictions"
    predictions_directory.mkdir()
    observations = base.read_observations(data_dir / base_config["source_observations"]["filename"])
    functional_features = build_public_functional_features(
        observations,
        ridge=float(base_config["model"]["spline_ridge"]),
        change_hours=tuple(map(int, base_config["model"]["change_hours"])),
    )
    anchor_path = ROOT / base_config["immutable_inputs"]["alpha50_proxy"]["path"]
    outputs: dict[str, Any] = {}
    veto_receipts: dict[str, Any] = {}
    inner_truth_rows = 0
    veto = config["veto"]
    for fold_name, fold_spec in base_config["folds"].items():
        inner = pd.concat(
            [
                inner_oof_block(
                    outer_name=fold_name,
                    outer_spec=fold_spec,
                    held_block=held_block,
                    base_config=base_config,
                    observations=observations,
                    functional_features=functional_features,
                    anchor_path=anchor_path,
                )
                for held_block in fold_spec["training_blocks"]
            ],
            ignore_index=True,
        )
        inner_truth_rows += len(inner)
        frozen = read_frozen_prediction(config["base_experiment"]["predictions"][fold_name])
        query_times = pd.DatetimeIndex(sorted(frozen["time"].unique()))
        decisions, receipts = trainonly_regime_decisions(
            inner,
            query_times,
            bin_days=int(veto["season_bin_days"]),
            window_days=float(veto["season_window_days"]),
            minimum_source_blocks=int(veto["minimum_source_blocks"]),
            minimum_profiles=int(veto["minimum_profiles"]),
            minimum_kst_days=int(veto["minimum_kst_days"]),
            bootstrap_replicates=int(veto["bootstrap_replicates"]),
            bootstrap_seed=int(veto["bootstrap_seed"]),
            ci90_upper_below=float(veto["ci90_upper_below_c"]),
        )
        bins = season_bin(frozen["time"], int(veto["season_bin_days"]))
        regime_enabled = np.asarray([decisions[int(value)] for value in bins], dtype=bool)
        final_enabled = frozen["original_enabled"].to_numpy(dtype=bool) & regime_enabled
        correction = np.where(final_enabled, frozen["original_correction"].to_numpy(), 0.0)
        frozen["regime_enabled"] = regime_enabled
        frozen["final_enabled"] = final_enabled
        frozen["correction"] = correction
        frozen["candidate"] = frozen["reference"].to_numpy() + correction
        if np.max(np.abs(correction[~regime_enabled]), initial=0.0) > 1e-12:
            raise base.ContractError("veto-disabled correction is nonzero")
        if np.max(
            np.abs(
                correction[final_enabled]
                - frozen.loc[final_enabled, "original_correction"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        ) > 1e-12:
            raise base.ContractError("enabled correction vector changed")
        path = predictions_directory / f"{fold_name}.npz"
        write_prediction(path, frozen)
        outputs[fold_name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(frozen),
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        veto_receipts[fold_name] = {
            "inner_truth_rows": len(inner),
            "inner_source_blocks": sorted(inner["source_block"].unique().tolist()),
            "bins": receipts,
            "regime_enabled_profile_share": float(
                pd.DataFrame({"time": frozen["time"], "enabled": regime_enabled})
                .drop_duplicates("time")["enabled"]
                .mean()
            ),
        }
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "truth_metric_computed": False,
        "outer_validation_truth_column_loaded": False,
        "inner_oof_truth_rows": inner_truth_rows,
        "correction_vector_modified": False,
        "official_rows_read": 0,
        "outputs": outputs,
        "veto_receipts": veto_receipts,
        "config_sha256": base.sha256_file(CONFIG),
    }
    base.atomic_json(output_directory / "prediction_commitment.json", commitment)

    scored_frames: list[pd.DataFrame] = []
    for fold_name, record in outputs.items():
        path = ROOT / record["path"]
        if base.sha256_file(path) != record["sha256"]:
            raise base.ContractError("committed prediction changed")
        with np.load(path, allow_pickle=False) as payload:
            scored = pd.DataFrame(
                {
                    "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                    "layer": payload["layer"].astype(int),
                    "reference": payload["reference"],
                    "candidate": payload["candidate"],
                    "correction": payload["correction"],
                    "original_enabled": payload["original_enabled"],
                    "regime_enabled": payload["regime_enabled"],
                    "final_enabled": payload["final_enabled"],
                }
            )
        truth = base.block_anchor(anchor_path, fold_name, include_truth=True)[["time", "layer", "truth"]]
        scored = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != int(record["rows"]):
            raise base.ContractError("outer truth binding failed")
        scored["fold"] = fold_name
        scored_frames.append(scored)
    scored = pd.concat(scored_frames, ignore_index=True)
    metrics = {
        "aggregate": base.metric_record(scored),
        "by_fold": {
            str(key): base.metric_record(group) for key, group in scored.groupby("fold", sort=True)
        },
        "by_layer": {
            str(int(key)): base.metric_record(group) for key, group in scored.groupby("layer", sort=True)
        },
    }
    gate = base_config["gate"]
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(gate["bootstrap_replicates"]),
        seed=int(gate["bootstrap_seed"]),
    )
    fold_deltas = [record["delta_rmse"] for record in metrics["by_fold"].values()]
    layer_deltas = [record["delta_rmse"] for record in metrics["by_layer"].values()]
    correction = scored["correction"].to_numpy(dtype=np.float64)
    active_share = float(np.mean(np.abs(correction) > 1e-12))
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    checks = {
        "pooled_delta": metrics["aggregate"]["delta_rmse"] <= float(gate["pooled_delta_rmse_max_c"]),
        "bootstrap_ci": bootstrap["ci90_high"] < float(gate["bootstrap_ci90_upper_max_c"]),
        "2024_sep_oct": metrics["by_fold"]["2024_sep_oct"]["delta_rmse"] <= float(gate["2024_sep_oct_delta_rmse_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas) >= int(gate["minimum_improved_folds"]),
        "worst_fold": max(fold_deltas) <= float(gate["maximum_worst_fold_regression_c"]),
        "worst_layer": max(layer_deltas) <= float(gate["maximum_layer_regression_c"]),
        "active_share": float(gate["minimum_active_share"]) <= active_share <= float(gate["maximum_active_share"]),
        "correction_rms": float(gate["minimum_correction_rms_c"]) <= correction_rms <= float(gate["maximum_correction_rms_c"]),
        "correction_p99": correction_p99 <= float(gate["maximum_correction_p99_c"]),
        "veto_disabled_exact_noop": float(np.max(np.abs(scored.loc[~scored["regime_enabled"], "correction"]), initial=0.0)) <= 1e-12,
        "correction_vector_unchanged": True,
    }
    result = {
        "schema_version": "p2.trainonly_regime_veto.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY_NO_UPLOAD" if all(checks.values()) else "NO_GO_EXACT_NO_OUTPUT",
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "metrics": metrics,
        "bootstrap": bootstrap,
        "active_share": active_share,
        "correction_rms_c": correction_rms,
        "correction_p99_c": correction_p99,
        "gate_checks": checks,
        "inner_truth_rows_read_before_commitment": inner_truth_rows,
        "outer_truth_rows_read_after_commitment": len(scored),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version()},
    }
    base.atomic_json(output_directory / "result.json", result)
    scored.drop(columns="truth").to_parquet(output_directory / "scored_predictions_no_truth.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = arguments.data_dir.expanduser().resolve()
    config, base_config = load_config(data_dir)
    if arguments.check:
        print(
            json.dumps(
                {"experiment_id": EXPERIMENT_ID, "status": "PASS", "official_rows_read": 0},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(run(config, base_config, data_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

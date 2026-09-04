"""Run the sealed P2 three-role cross-fit regime-veto experiment exactly once."""

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
    paired_kst_day_bootstrap,
)
from p2_restore.p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2 import (  # noqa: E402
    contiguous_time_groups,
    time_group_sha256,
)
from p2_restore.profile_projection import public_endpoint_frame  # noqa: E402
from p2_restore.supervised_rank1_functional_residual import (  # noqa: E402
    TARGET_LAYERS,
    SupervisedRank1Residual,
    build_public_functional_features,
)
from p2_restore.trainonly_regime_veto import season_bin, trainonly_regime_decisions  # noqa: E402
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_trainonly_regime_veto_20260828_v1 as old_veto,
)

EXPERIMENT_ID = "p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2"
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
            policy["closed_v1_contract_reuse_or_relaxation_allowed"],
        )
    ):
        raise base.ContractError("forbidden official access, retry, or v1 relaxation enabled")
    records = [config["base_experiment"][key] for key in ("config", "runner", "result", "commitment")]
    records.extend(config["base_experiment"]["predictions"].values())
    for record in records:
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise base.ContractError(f"immutable input changed: {path}")
    base_config = json.loads(
        (ROOT / config["base_experiment"]["config"]["path"]).read_text(encoding="utf-8")
    )
    observations = data_dir / base_config["source_observations"]["filename"]
    if (
        not observations.is_file()
        or base.sha256_file(observations) != base_config["source_observations"]["sha256"]
    ):
        raise base.ContractError("observations.csv pin changed")
    if config["crossfit"]["rotations"] != [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        raise base.ContractError("three-role rotation drifted")
    return config, base_config


def _training_anchor_keys(
    anchor_path: Path,
    blocks: list[str],
) -> pd.DataFrame:
    frame = pd.concat(
        [base.block_anchor(anchor_path, block, include_truth=False)[["time", "layer"]] for block in blocks],
        ignore_index=True,
    )
    counts = frame.groupby("time", sort=True)["layer"].nunique()
    complete = pd.DatetimeIndex(counts[counts == len(TARGET_LAYERS)].index)
    if complete.empty:
        raise base.ContractError("outer training has no complete target profiles")
    return frame.loc[frame["time"].isin(complete)].sort_values(["time", "layer"]).reset_index(drop=True)


def _group_anchor(
    anchor_path: Path,
    blocks: list[str],
    group_times: pd.DatetimeIndex,
    *,
    include_truth: bool,
) -> pd.DataFrame:
    frame = pd.concat(
        [base.block_anchor(anchor_path, block, include_truth=include_truth) for block in blocks],
        ignore_index=True,
    )
    result = frame.loc[frame["time"].isin(group_times)].sort_values(["time", "layer"])
    if result.empty or result["time"].nunique() != len(group_times):
        raise base.ContractError("cross-fit group anchor coverage failed")
    return result.reset_index(drop=True)


def _role_prediction(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    held_group: int,
    fit_group: int,
    support_group: int,
    groups: tuple[pd.DatetimeIndex, ...],
    base_config: dict[str, Any],
    observations: pd.DataFrame,
    functional_features: pd.DataFrame,
    anchor_path: Path,
) -> pd.DataFrame:
    """Predict H using a correction fit on M and a reference fit only on disjoint R."""

    outer_start = base.utc(outer_spec["start"])
    blocks = list(outer_spec["training_blocks"])
    held_times, fit_times, support_times = (
        groups[held_group],
        groups[fit_group],
        groups[support_group],
    )
    if set(held_times.asi8) & set(fit_times.asi8) or set(held_times.asi8) & set(support_times.asi8):
        raise base.ContractError("cross-fit role overlap")
    masked = observations.copy()
    target = masked["layer"].isin(TARGET_LAYERS)
    support = masked["time"].isin(support_times)
    masked.loc[target & ~support, ["temp", "psal"]] = np.nan
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)

    held = base.add_metadata(
        _group_anchor(anchor_path, blocks, held_times, include_truth=False), observations
    )
    held_reference, _ = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=held,
        train_stop=outer_start,
        config=base_config,
    )
    training = base.add_metadata(
        _group_anchor(anchor_path, blocks, fit_times, include_truth=True), observations
    )
    if not training["time"].lt(outer_start).all():
        raise base.ContractError("correction-fit labels cross outer boundary")
    training_reference, _ = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=training,
        train_stop=outer_start,
        config=base_config,
    )
    training["residual"] = training["truth"].to_numpy(dtype=np.float64) - training_reference
    train_times, response = base.profile_response(training)
    train_features = base.align_features(functional_features, train_times)
    valid = train_features["public_profile_valid"].to_numpy(dtype=bool)
    fitted = SupervisedRank1Residual.fit(
        train_features.loc[valid].reset_index(drop=True), response[valid], train_times[valid]
    )
    held_profile_times = pd.DatetimeIndex(sorted(held["time"].unique()))
    prediction = old_veto._profile_prediction(
        query=held,
        query_features=base.align_features(functional_features, held_profile_times),
        fitted=fitted,
        reference=held_reference,
        endpoints=endpoints,
        model_config=base_config["model"],
    )
    # Truth binding is deliberately last: H has influenced neither R reference nor M correction fit.
    truth = _group_anchor(anchor_path, blocks, held_times, include_truth=True)[
        ["time", "layer", "truth"]
    ]
    prediction = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
    if prediction["truth"].isna().any():
        raise base.ContractError("held truth binding failed")
    prediction["source_block"] = f"group_{held_group}"
    prediction["outer_fold"] = outer_name
    prediction["reference_support_group"] = support_group
    prediction["correction_fit_group"] = fit_group
    return prediction


def _write_prediction(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8,
        layer=frame["layer"].to_numpy(dtype=np.int16),
        reference=frame["reference"].to_numpy(dtype=np.float64),
        candidate=frame["candidate"].to_numpy(dtype=np.float64),
        correction=frame["correction"].to_numpy(dtype=np.float64),
        original_correction=frame["original_correction"].to_numpy(dtype=np.float64),
        original_enabled=frame["original_enabled"].to_numpy(dtype=bool),
        regime_enabled=frame["regime_enabled"].to_numpy(dtype=bool),
        final_enabled=frame["final_enabled"].to_numpy(dtype=bool),
    )


def run(config: dict[str, Any], base_config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = ROOT / config["artifact_directory"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    observations = base.read_observations(data_dir / base_config["source_observations"]["filename"])
    functional_features = build_public_functional_features(
        observations,
        ridge=float(base_config["model"]["spline_ridge"]),
        change_hours=tuple(map(int, base_config["model"]["change_hours"])),
    )
    anchor_path = ROOT / base_config["immutable_inputs"]["alpha50_proxy"]["path"]
    outputs: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    inner_truth_rows = 0
    crossfit = config["crossfit"]
    for fold_name, fold_spec in base_config["folds"].items():
        keys = _training_anchor_keys(anchor_path, list(fold_spec["training_blocks"]))
        groups = contiguous_time_groups(
            keys["time"],
            groups=int(crossfit["time_groups"]),
            minimum_profiles=int(crossfit["minimum_group_profiles"]),
        )
        group_receipt = [
            {
                "group": index,
                "profiles": len(group),
                "start": group[0].isoformat(),
                "stop_inclusive": group[-1].isoformat(),
                "time_sha256": time_group_sha256(group),
            }
            for index, group in enumerate(groups)
        ]
        inner_parts = []
        for held_group, fit_group, support_group in crossfit["rotations"]:
            inner_parts.append(
                _role_prediction(
                    outer_name=fold_name,
                    outer_spec=fold_spec,
                    held_group=int(held_group),
                    fit_group=int(fit_group),
                    support_group=int(support_group),
                    groups=groups,
                    base_config=base_config,
                    observations=observations,
                    functional_features=functional_features,
                    anchor_path=anchor_path,
                )
            )
        inner = pd.concat(inner_parts, ignore_index=True)
        inner_truth_rows += len(inner)
        frozen = old_veto.read_frozen_prediction(config["base_experiment"]["predictions"][fold_name])
        query_times = pd.DatetimeIndex(sorted(frozen["time"].unique()))
        decisions, bins_receipt = trainonly_regime_decisions(
            inner,
            query_times,
            bin_days=int(crossfit["season_bin_days"]),
            window_days=float(crossfit["season_window_days"]),
            minimum_source_blocks=int(crossfit["minimum_source_groups"]),
            minimum_profiles=int(crossfit["minimum_profiles"]),
            minimum_kst_days=int(crossfit["minimum_kst_days"]),
            bootstrap_replicates=int(crossfit["bootstrap_replicates"]),
            bootstrap_seed=int(crossfit["bootstrap_seed"]),
            ci90_upper_below=float(crossfit["ci90_upper_below_c"]),
        )
        bins = season_bin(frozen["time"], int(crossfit["season_bin_days"]))
        regime_enabled = np.asarray([decisions[int(value)] for value in bins], dtype=bool)
        final_enabled = frozen["original_enabled"].to_numpy(dtype=bool) & regime_enabled
        original_correction = frozen["original_correction"].to_numpy(dtype=np.float64)
        correction = np.where(final_enabled, original_correction, 0.0)
        frozen["regime_enabled"] = regime_enabled
        frozen["final_enabled"] = final_enabled
        frozen["original_correction"] = original_correction
        frozen["correction"] = correction
        frozen["candidate"] = frozen["reference"].to_numpy(dtype=np.float64) + correction
        if np.max(np.abs(correction[~regime_enabled]), initial=0.0) > 1e-12:
            raise base.ContractError("veto-disabled row is not exact reference no-op")
        if np.max(np.abs(correction[final_enabled] - original_correction[final_enabled]), initial=0.0) > 1e-12:
            raise base.ContractError("enabled sealed correction was modified")
        path = predictions_dir / f"{fold_name}.npz"
        _write_prediction(path, frozen)
        outputs[fold_name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(frozen),
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        receipts[fold_name] = {
            "groups": group_receipt,
            "inner_truth_rows": len(inner),
            "inner_source_groups": sorted(inner["source_block"].unique().tolist()),
            "season_bins": bins_receipt,
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
        "inner_truth_rows": inner_truth_rows,
        "crossfit_roles": ["held_label_late", "correction_fit", "reference_support"],
        "correction_vector_modified": False,
        "official_rows_read": 0,
        "outputs": outputs,
        "crossfit_receipts": receipts,
        "config_sha256": base.sha256_file(CONFIG),
    }
    base.atomic_json(output / "prediction_commitment.json", commitment)

    scored_parts = []
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
                    "original_correction": payload["original_correction"],
                    "regime_enabled": payload["regime_enabled"],
                }
            )
        truth = base.block_anchor(anchor_path, fold_name, include_truth=True)[["time", "layer", "truth"]]
        scored = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != int(record["rows"]):
            raise base.ContractError("outer truth binding failed")
        scored["fold"] = fold_name
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    metrics = {
        "aggregate": base.metric_record(scored),
        "by_fold": {key: base.metric_record(group) for key, group in scored.groupby("fold", sort=True)},
        "by_layer": {
            str(int(key)): base.metric_record(group) for key, group in scored.groupby("layer", sort=True)
        },
    }
    gate = config["gate"]
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(gate["bootstrap_replicates"]),
        seed=int(gate["bootstrap_seed"]),
    )
    fold_deltas = [item["delta_rmse"] for item in metrics["by_fold"].values()]
    layer_deltas = [item["delta_rmse"] for item in metrics["by_layer"].values()]
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
        "veto_disabled_exact_noop": float(
            np.max(
                np.abs(
                    scored.loc[~scored["regime_enabled"], "correction"].to_numpy(
                        dtype=np.float64
                    )
                ),
                initial=0.0,
            )
        )
        <= 1e-12,
        "enabled_correction_vector_unchanged": float(
            np.max(
                np.abs(
                    scored.loc[scored["regime_enabled"], "correction"].to_numpy(
                        dtype=np.float64
                    )
                    - scored.loc[
                        scored["regime_enabled"], "original_correction"
                    ].to_numpy(dtype=np.float64)
                ),
                initial=0.0,
            )
        )
        <= 1e-12,
    }
    result = {
        "schema_version": "p2.threeway_crossfit_regime_veto.result.v2",
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
    base.atomic_json(output / "result.json", result)
    scored.drop(columns="truth").to_parquet(output / "scored_predictions_no_truth.parquet", index=False)
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
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "PASS", "official_rows_read": 0}, indent=2))
        return
    print(json.dumps(run(config, base_config, data_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

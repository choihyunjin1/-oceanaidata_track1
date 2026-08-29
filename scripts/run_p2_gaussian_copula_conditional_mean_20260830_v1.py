"""Run the sealed train-only P2 Gaussian-copula conditional-mean pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel  # noqa: E402
from p2_restore.gaussian_copula_conditional_mean import (  # noqa: E402
    CopulaContractError,
    SeasonalCopulaConditionalMean,
)
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    bounded_profile_correction,
    paired_kst_day_bootstrap,
    rmse,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.supervised_rank1_functional_residual import TARGET_LAYERS  # noqa: E402
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)

EXPERIMENT_ID = "p2_gaussian_copula_conditional_mean_20260830_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reference_config(config: dict[str, Any]) -> dict[str, Any]:
    reference = config["reference"]
    return {
        "model": {
            "reference_alpha": float(reference["alpha"]),
            "season_bin_days": int(reference["season_bin_days"]),
            "season_window_days": float(reference["season_window_days"]),
            "minimum_season_rows": int(reference["minimum_season_rows"]),
            "fallback_nearest_complete_rows": int(
                reference["fallback_nearest_complete_rows"]
            ),
            "spline_ridge": float(reference["spline_ridge"]),
            "change_hours": list(reference["change_hours"]),
        }
    }


def load_config(data_dir: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise base.ContractError("experiment ID drifted")
    policy = config["execution_policy"]
    forbidden = (
        policy["official_hidden_gap_values_read_allowed"],
        policy["official_test_sample_submission_read_allowed"],
        policy["submission_csv_generation_allowed"],
        policy["official_upload_authorized"],
        policy["result_based_retry"],
        policy["wide_hpo_allowed"],
    )
    if any(forbidden) or int(policy["maximum_executions"]) != 1:
        raise base.ContractError("forbidden access, HPO, retry, or execution policy drifted")
    if list(map(float, config["copula"]["shrinkage_candidates"])) != [0.1, 0.3, 0.5]:
        raise base.ContractError("sealed shrinkage set drifted")
    if int(config["copula"]["inner_time_groups"]) != 3 or len(config["folds"]) != 3:
        raise base.ContractError("three-by-three validation contract drifted")
    if float(config["reference"]["alpha"]) != 0.5:
        raise base.ContractError("alpha50 comparator drifted")
    if int(config["resource_contract"]["maximum_conceptual_copula_fits"]) != 30:
        raise base.ContractError("fit budget drifted")
    for record in config["immutable_inputs"].values():
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise base.ContractError(f"immutable input changed: {path}")
    observations = data_dir / config["source_observations"]["filename"]
    source = config["source_observations"]
    if (
        not observations.is_file()
        or observations.stat().st_size != int(source["bytes"])
        or base.sha256_file(observations) != source["sha256"]
    ):
        raise base.ContractError("observations.csv pin changed")
    support = json.loads(
        (ROOT / config["immutable_inputs"]["train_only_support_audit"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if support.get("status") != "TRAIN_ONLY_SUPPORT_PASS_QUERY_AUDIT_NOT_AUTHORIZED":
        raise base.ContractError("train-only support audit no longer passes")
    return config


def season_labels(times: pd.DatetimeIndex, config: dict[str, Any]) -> np.ndarray:
    month_to_season = {
        int(month): season
        for season, months in config["copula"]["season_definition"].items()
        for month in months
    }
    labels = np.asarray([month_to_season[int(month)] for month in times.month], dtype=str)
    if len(labels) != len(times):
        raise base.ContractError("season labels do not align")
    return labels


def profile_design(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_response: bool,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray | None, np.ndarray]:
    required = {"time", "layer", "current_blend50", "reference"}
    if require_response:
        required.add("residual")
    if not required.issubset(frame.columns):
        raise base.ContractError("profile design columns changed")
    if frame.duplicated(["time", "layer"]).any():
        raise base.ContractError("profile design keys duplicate")
    columns = list(TARGET_LAYERS)
    current = frame.pivot(index="time", columns="layer", values="current_blend50").reindex(
        columns=columns
    )
    reference = frame.pivot(index="time", columns="layer", values="reference").reindex(
        columns=columns
    )
    complete = current.notna().all(axis=1) & reference.notna().all(axis=1)
    response_pivot: pd.DataFrame | None = None
    if require_response:
        response_pivot = frame.pivot(index="time", columns="layer", values="residual").reindex(
            columns=columns
        )
        complete &= response_pivot.notna().all(axis=1)
    times = pd.DatetimeIndex(current.index[complete])
    x = np.column_stack(
        [
            current.loc[complete].to_numpy(dtype=np.float64),
            (reference.loc[complete] - current.loc[complete]).to_numpy(dtype=np.float64),
        ]
    )
    y = (
        None
        if response_pivot is None
        else response_pivot.loc[complete].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(x).all() or (y is not None and not np.isfinite(y).all()):
        raise base.ContractError("profile design became nonfinite")
    return times, x, y, season_labels(times, config)


def fitted_copula(
    x: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    shrinkage: float,
    config: dict[str, Any],
) -> SeasonalCopulaConditionalMean:
    model = config["copula"]
    return SeasonalCopulaConditionalMean.fit(
        x,
        y,
        seasons,
        shrinkage=float(shrinkage),
        quadrature_order=int(model["quadrature_order"]),
        eigen_floor=float(model["eigenvalue_floor"]),
        maximum_condition_number=float(model["maximum_condition_number"]),
        minimum_season_profiles=int(model["minimum_season_profiles"]),
    )


def select_shrinkage(
    times: pd.DatetimeIndex,
    x: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    config: dict[str, Any],
    fit_counter: dict[str, int],
) -> tuple[float, list[dict[str, Any]], bool]:
    order = np.argsort(times.asi8)
    groups = np.empty(len(times), dtype=np.int8)
    for group, indices in enumerate(np.array_split(order, int(config["copula"]["inner_time_groups"]))):
        if len(indices) < int(config["copula"]["minimum_inner_group_profiles"]):
            raise base.ContractError("inner group lacks minimum profile support")
        groups[indices] = group
    records: list[dict[str, Any]] = []
    for shrinkage in map(float, config["copula"]["shrinkage_candidates"]):
        squared_reference = 0.0
        squared_candidate = 0.0
        rows = 0
        group_deltas: list[float] = []
        receipts: list[dict[str, Any]] = []
        for held_group in range(int(config["copula"]["inner_time_groups"])):
            held = groups == held_group
            train = ~held
            model = fitted_copula(x[train], y[train], seasons[train], shrinkage, config)
            fit_counter["inner_copula_fits"] += 1
            prediction = model.predict(x[held], seasons[held])
            reference_se = float(np.square(y[held]).sum())
            candidate_se = float(np.square(prediction - y[held]).sum())
            count = int(y[held].size)
            squared_reference += reference_se
            squared_candidate += candidate_se
            rows += count
            group_deltas.append(np.sqrt(candidate_se / count) - np.sqrt(reference_se / count))
            receipts.append(
                {
                    "held_group": held_group,
                    "train_profiles": int(train.sum()),
                    "held_profiles": int(held.sum()),
                    "model": model.receipt(),
                }
            )
        reference_rmse = float(np.sqrt(squared_reference / rows))
        candidate_rmse = float(np.sqrt(squared_candidate / rows))
        eligibility = config["copula"]["inner_eligibility"]
        eligible = (
            candidate_rmse - reference_rmse < float(eligibility["pooled_delta_lt_c"])
            and max(group_deltas)
            <= float(eligibility["maximum_worst_group_regression_c"])
        )
        records.append(
            {
                "shrinkage": shrinkage,
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
                "group_deltas": group_deltas,
                "eligible": bool(eligible),
                "fit_receipts": receipts,
            }
        )
    selected = min(
        records,
        key=lambda item: (
            item["candidate_rmse"],
            abs(float(item["shrinkage"]) - 0.3),
            float(item["shrinkage"]),
        ),
    )
    return float(selected["shrinkage"]), records, bool(selected["eligible"])


def row_correction(
    query: pd.DataFrame,
    query_times: pd.DatetimeIndex,
    profile_prediction: np.ndarray,
) -> np.ndarray:
    time_to_row = {pd.Timestamp(value): row for row, value in enumerate(query_times)}
    layer_to_column = {int(layer): column for column, layer in enumerate(TARGET_LAYERS)}
    result = np.empty(len(query), dtype=np.float64)
    for row, value in enumerate(query.itertuples(index=False)):
        result[row] = profile_prediction[time_to_row[pd.Timestamp(value.time)], layer_to_column[int(value.layer)]]
    return result


def fit_outer_prediction(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    config: dict[str, Any],
    observations: pd.DataFrame,
    anchor_path: Path,
    fit_counter: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    start, stop = base.utc(outer_spec["start"]), base.utc(outer_spec["stop"])
    masked = observations.copy()
    validation_mask = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(TARGET_LAYERS)
    )
    masked.loc[validation_mask, ["temp", "psal"]] = np.nan
    if not masked.loc[validation_mask, ["temp", "psal"]].isna().all().all():
        raise base.ContractError("joint outer target mask failed")
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    query = base.add_metadata(
        base.block_anchor(anchor_path, outer_name, include_truth=False), observations
    )
    model_config = reference_config(config)
    reference, reference_receipts = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=start,
        config=model_config,
    )
    query["reference"] = reference
    training_parts: list[pd.DataFrame] = []
    training_reference_receipts: dict[str, Any] = {}
    for training_block in outer_spec["training_blocks"]:
        training = base.add_metadata(
            base.block_anchor(anchor_path, training_block, include_truth=True), observations
        )
        if not training["time"].lt(start).all():
            raise base.ContractError("training label crosses outer boundary")
        bounds = config["block_bounds"][training_block]
        training_reference, receipts = base.alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=model_config,
            exclude=(base.utc(bounds[0]), base.utc(bounds[1])),
        )
        training["reference"] = training_reference
        training["residual"] = training["truth"].to_numpy(dtype=np.float64) - training_reference
        training_parts.append(training)
        training_reference_receipts[training_block] = receipts
    training = pd.concat(training_parts, ignore_index=True)
    train_times, train_x, train_y, train_seasons = profile_design(
        training, config, require_response=True
    )
    if train_y is None:
        raise base.ContractError("training residual matrix missing")
    selected, selection_records, inner_eligible = select_shrinkage(
        train_times, train_x, train_y, train_seasons, config, fit_counter
    )
    model = fitted_copula(train_x, train_y, train_seasons, selected, config)
    fit_counter["outer_copula_fits"] += 1
    query_times, query_x, _, query_seasons = profile_design(
        query, config, require_response=False
    )
    profile_prediction = model.predict(query_x, query_seasons)
    raw = row_correction(query, query_times, profile_prediction)
    bounded, cap_receipt = bounded_profile_correction(
        raw,
        np.ones(len(raw), dtype=bool),
        rms_cap=float(config["copula"]["correction_rms_cap_c"]),
        p99_cap=float(config["copula"]["correction_p99_cap_c"]),
    )
    candidate = project_profiles_vectorized(query, reference + bounded, endpoints).prediction
    correction = candidate - reference
    if not np.isfinite(candidate).all():
        raise base.ContractError("outer copula prediction became nonfinite")
    output = query[["time", "layer", "current_blend50"]].copy()
    output["reference"] = reference
    output["candidate"] = candidate
    output["correction"] = correction
    selection = {
        "selected_shrinkage": selected,
        "selected_inner_eligible": inner_eligible,
        "records": selection_records,
        "records_sha256": canonical_sha256(selection_records),
        "outer_truth_rows_read_before_selection": 0,
    }
    receipt = {
        "fold": outer_name,
        "outer_validation_truth_column_loaded": False,
        "validation_target_temp_psal_masked_together": True,
        "training_blocks": list(outer_spec["training_blocks"]),
        "training_profiles": int(len(train_times)),
        "query_profiles": int(len(query_times)),
        "model": model.receipt(),
        "reference_oas": reference_receipts,
        "training_reference_oas": training_reference_receipts,
        "preprojection_cap": cap_receipt,
        "postprojection_correction_rms_c": float(np.sqrt(np.mean(np.square(correction)))),
        "postprojection_correction_p99_c": float(np.quantile(np.abs(correction), 0.99)),
    }
    return output, selection, receipt


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8,
        layer=frame["layer"].to_numpy(dtype=np.int16),
        current_blend50=frame["current_blend50"].to_numpy(dtype=np.float64),
        reference=frame["reference"].to_numpy(dtype=np.float64),
        candidate=frame["candidate"].to_numpy(dtype=np.float64),
        correction=frame["correction"].to_numpy(dtype=np.float64),
    )


def read_prediction(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as payload:
        return pd.DataFrame(
            {
                "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                "layer": payload["layer"].astype(int),
                "current_blend50": payload["current_blend50"],
                "reference": payload["reference"],
                "candidate": payload["candidate"],
                "correction": payload["correction"],
            }
        )


def metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {
        "rows": int(len(frame)),
        "reference_rmse": reference,
        "candidate_rmse": candidate,
        "delta_rmse": candidate - reference,
    }


def report_markdown(result: dict[str, Any], commitment: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Pooled delta RMSE: `{metrics['aggregate']['delta_rmse']:.9f}` °C",
        f"- Paired KST-day bootstrap upper q95: `{result['bootstrap']['ci90_high']:.9f}` °C",
        f"- Conceptual copula fits: `{result['fit_counts']['total_copula_fits']}`",
        f"- Runtime: `{result['runtime']['elapsed_seconds']:.3f}` seconds",
        "- Official hidden/test/sample/submission rows read: `0`; CSV/upload: `false`.",
        "",
        "## Outer selections",
        "",
    ]
    for fold, selection in commitment["selections"].items():
        lines.append(
            f"- `{fold}`: shrinkage `{selection['selected_shrinkage']}`; "
            f"inner eligible `{selection['selected_inner_eligible']}`"
        )
    lines.extend(["", "## Metrics by fold", ""])
    for fold, value in metrics["by_fold"].items():
        lines.append(f"- `{fold}`: ΔRMSE `{value['delta_rmse']:.9f}` °C")
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- `{name}`: `{passed}`" for name, passed in result["gate_checks"].items())
    lines.extend(
        [
            "",
            "This closes only the exact seasonal empirical-margin/Kendall-correlation/shrinkage recipe. "
            "All outer blocks are historically exposed, so a pass would remain research-only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = ROOT / config["artifact_directory"]
    report_directory = ROOT / config["report_directory"]
    if output.exists() or report_directory.exists():
        raise FileExistsError("one-shot output already exists")
    output.mkdir(parents=True)
    predictions_directory = output / "predictions"
    predictions_directory.mkdir()
    atomic_json(
        output / "attempt.lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "maximum_executions": 1,
            "config_sha256": base.sha256_file(CONFIG),
            "result_based_retry": False,
        },
    )
    observations = base.read_observations(
        data_dir / config["source_observations"]["filename"]
    )
    anchor_path = ROOT / config["immutable_inputs"]["alpha50_proxy"]["path"]
    outputs: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    fit_counter = {"inner_copula_fits": 0, "outer_copula_fits": 0}
    with threadpool_limits(limits=int(config["resource_contract"]["blas_threads"])):
        for outer_name, outer_spec in config["folds"].items():
            prediction, selection, receipt = fit_outer_prediction(
                outer_name=outer_name,
                outer_spec=outer_spec,
                config=config,
                observations=observations,
                anchor_path=anchor_path,
                fit_counter=fit_counter,
            )
            path = predictions_directory / f"{outer_name}.npz"
            write_prediction(path, prediction)
            outputs[outer_name] = {
                "path": str(path.relative_to(ROOT)),
                "rows": int(len(prediction)),
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            }
            selections[outer_name] = selection
            receipts[outer_name] = receipt
    total_fits = fit_counter["inner_copula_fits"] + fit_counter["outer_copula_fits"]
    if total_fits != int(config["resource_contract"]["maximum_conceptual_copula_fits"]):
        raise base.ContractError("conceptual fit count drifted")
    commitment = {
        "schema_version": "p2.gaussian_copula.prediction_commitment.v1",
        "experiment_id": EXPERIMENT_ID,
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "truth_metric_computed": False,
        "outer_validation_truth_rows_read": 0,
        "official_rows_read": 0,
        "outputs": outputs,
        "selections": selections,
        "receipts": receipts,
        "fit_counts": {**fit_counter, "total_copula_fits": total_fits},
        "config_sha256": base.sha256_file(CONFIG),
        "observations_sha256": config["source_observations"]["sha256"],
        "anchor_sha256": config["immutable_inputs"]["alpha50_proxy"]["sha256"],
    }
    atomic_json(output / "prediction_commitment.json", commitment)

    scored_frames: list[pd.DataFrame] = []
    for fold, record in outputs.items():
        path = ROOT / record["path"]
        if base.sha256_file(path) != record["sha256"]:
            raise base.ContractError("sealed prediction changed")
        prediction = read_prediction(path)
        truth = base.block_anchor(anchor_path, fold, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != int(record["rows"]):
            raise base.ContractError("late truth binding failed")
        scored["fold"] = fold
        scored_frames.append(scored)
    scored = pd.concat(scored_frames, ignore_index=True)
    metrics = {
        "aggregate": metric_record(scored),
        "by_fold": {
            str(key): metric_record(group) for key, group in scored.groupby("fold", sort=True)
        },
        "by_layer": {
            str(int(key)): metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["gate"]["bootstrap_replicates"]),
        seed=int(config["gate"]["bootstrap_seed"]),
    )
    fold_deltas = [float(value["delta_rmse"]) for value in metrics["by_fold"].values()]
    layer_deltas = [float(value["delta_rmse"]) for value in metrics["by_layer"].values()]
    correction = scored["correction"].to_numpy(dtype=np.float64)
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    gate = config["gate"]
    checks = {
        "structural_copula_contract": True,
        "all_inner_selections_eligible": all(
            bool(value["selected_inner_eligible"]) for value in selections.values()
        ),
        "pooled_delta_rmse_lt_0": metrics["aggregate"]["delta_rmse"]
        < float(gate["pooled_delta_rmse_max_c"]),
        "at_least_two_of_three_folds_improve": sum(value < 0.0 for value in fold_deltas)
        >= int(gate["minimum_improved_folds"]),
        "no_fold_regresses": max(fold_deltas)
        <= float(gate["maximum_worst_fold_regression_c"]),
        "no_layer_worse_by_more_than_0_001_c": max(layer_deltas)
        <= float(gate["maximum_layer_regression_c"]),
        "paired_bootstrap_upper_lt_0": bootstrap["ci90_high"]
        < float(gate["bootstrap_upper_max_c"]),
        "correction_rms_cap": correction_rms
        <= float(gate["maximum_correction_rms_c"]) + 1e-12,
        "correction_p99_cap": correction_p99
        <= float(gate["maximum_correction_p99_c"]) + 1e-12,
    }
    result = {
        "schema_version": "p2.gaussian_copula_conditional_mean.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "GO_RESEARCH_ONLY_NO_UPLOAD"
            if all(checks.values())
            else "NO_GO_CLOSE_EXACT_COPULA_AXIS"
        ),
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "metrics": metrics,
        "bootstrap": bootstrap,
        "correction_rms_c": correction_rms,
        "correction_p99_c": correction_p99,
        "gate_checks": checks,
        "fit_counts": commitment["fit_counts"],
        "prediction_commitment_sha256": base.sha256_file(output / "prediction_commitment.json"),
        "truth_rows_read_after_commitment": int(len(scored)),
        "official_hidden_test_sample_submission_rows_read": 0,
        "submission_csv_count": 0,
        "upload_count": 0,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
        },
    }
    atomic_json(output / "result.json", result)
    scored.drop(columns="truth").to_parquet(
        output / "scored_predictions_no_truth.parquet", index=False
    )
    report_directory.mkdir(parents=True)
    (report_directory / "report-source.md").write_text(
        report_markdown(result, commitment), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check == args.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = args.data_dir.expanduser().resolve()
    config = load_config(data_dir)
    if args.check:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "READY_GUARDED",
                    "folds": len(config["folds"]),
                    "shrinkage_candidates": config["copula"]["shrinkage_candidates"],
                    "maximum_conceptual_copula_fits": config["resource_contract"][
                        "maximum_conceptual_copula_fits"
                    ],
                    "official_rows_read": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    try:
        result = run(config, data_dir)
    except CopulaContractError as error:
        raise base.ContractError(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

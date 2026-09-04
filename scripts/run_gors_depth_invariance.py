"""Run the authorized one-shot G-ORS deployment-depth invariance comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.config import P1QCConfig, load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.depth_shift_stress import (  # noqa: E402
    _fold_reference,
    apply_depth_missing_counterfactual,
    depth_fallback_codes,
)
from p1_qc.experiment import RunRecorder, sha256_file  # noqa: E402
from p1_qc.gors_depth_invariance import (  # noqa: E402
    GORS_STATION,
    encode_gors_depth_invariant_fold,
)
from p1_qc.gors_depth_invariance_preregistration import (  # noqa: E402
    EXACT_GATES,
    FOLD_POSTPROCESS,
    REFERENCE_HASHES,
    validate_gors_depth_preregistration_files,
)
from p1_qc.metrics import (  # noqa: E402
    binary_counts,
    evaluate_predictions,
    group_row_shares,
    weighted_group_counts,
)
from p1_qc.pipeline import (  # noqa: E402
    TabularEncoder,
    _fit_model,
    _iteration_parameter,
    _model_parameters,
    _threads,
    apply_postprocess,
    load_or_build_features,
    resolve_data_dir,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes  # noqa: E402
from p1_qc.validation import (  # noqa: E402
    normal_station_layer_day_fp,
    paired_block_bootstrap,
)

BACKEND = "xgboost"
ITERATIONS = 700
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260813
PROBABILITY_TOLERANCE = 1.0e-6
A_EXPECTED_WEIGHTED_F1 = 0.8043980282796417
A_EXPECTED_GORS_F1 = 0.7633410672853829
A_REPRODUCTION_TOLERANCE = 1.0e-12
KEY_COLUMNS = ("station", "year", "layer", "time")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p1_gors_depth_invariance_draft.json"),
    )
    parser.add_argument("--ledger", type=Path, default=Path("reports/EXPERIMENT_LEDGER.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("artifacts/runs/20260813T153038+0900_cv_378a4e89"),
    )
    parser.add_argument(
        "--deployment-stress",
        type=Path,
        default=Path("artifacts/depth_shift_stress_20260813/result.json"),
    )
    return parser.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _input_paths(
    *,
    config_path: Path,
    data_dir: Path,
    reference_run: Path,
    deployment_stress: Path,
) -> dict[str, Path]:
    return {
        "config": config_path,
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "oof": reference_run / "oof.parquet",
        "metrics": reference_run / "metrics.json",
        "selection": reference_run / "selection.json",
        "deployment_stress": deployment_stress,
    }


def _assert_existing_stress(path: Path) -> dict[str, Any]:
    payload = _json(path)
    try:
        aggregate = payload["scenarios"]["gors_depth_100pct_missing"]["aggregate"]
        weighted = float(aggregate["test_share_weighted_all_outer"]["counterfactual"]["f1"])
        gors = float(aggregate["affected"]["counterfactual"]["f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("deployment stress lacks the frozen A aggregate") from exc
    if abs(weighted - A_EXPECTED_WEIGHTED_F1) > A_REPRODUCTION_TOLERANCE:
        raise RuntimeError("frozen A weighted F1 differs from the preregistered value")
    if abs(gors - A_EXPECTED_GORS_F1) > A_REPRODUCTION_TOLERANCE:
        raise RuntimeError("frozen A G-ORS F1 differs from the preregistered value")
    return {
        "test_share_weighted_f1": weighted,
        "gors_f1": gors,
        "source_sha256": sha256_file(path),
    }


def _label_blind_fold_indices(
    train: pd.DataFrame,
    reference_oof: pd.DataFrame,
    config: P1QCConfig,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Restore frozen event-protected validation membership without label values."""

    required = {*KEY_COLUMNS, "fold"}
    missing = sorted(required.difference(reference_oof.columns))
    if missing:
        raise KeyError(f"reference OOF lacks fold membership columns: {missing}")
    source_keys = train.loc[:, list(KEY_COLUMNS)].copy()
    if source_keys.duplicated().any():
        raise RuntimeError("source keys are not unique")
    source_keys["__position"] = np.arange(len(train), dtype=np.int64)
    time = pd.to_datetime(train["time"], errors="raise", utc=True)
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec in config.splits.folds:
        reference = reference_oof.loc[reference_oof["fold"].eq(spec.name), list(KEY_COLUMNS)].copy()
        if reference.duplicated().any() or reference.empty:
            raise RuntimeError(f"{spec.name} frozen validation membership is invalid")
        mapped = reference.merge(
            source_keys,
            on=list(KEY_COLUMNS),
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if mapped["__position"].isna().any():
            raise RuntimeError(f"{spec.name} frozen keys are absent from train")
        val_idx = mapped["__position"].to_numpy(dtype=np.int64)
        # Merges normally preserve left order; explicitly verify because row
        # order must reproduce the frozen probability vector exactly.
        observed = train.iloc[val_idx].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        if not observed.equals(reference.reset_index(drop=True)):
            raise RuntimeError(f"{spec.name} validation key order was not reproduced")
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        train_idx = np.flatnonzero(time.le(train_end).to_numpy())
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError(f"{spec.name} train/validation overlap")
        results.append((spec.name, train_idx, val_idx))
    return results


def _arm_predictions_without_truth(
    train: pd.DataFrame,
    bundle: Any,
    config: P1QCConfig,
    reference_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build secondary/A/B predictions before reading any outer label."""

    folds = _label_blind_fold_indices(train, reference_oof, config)
    parameters = _model_parameters(config, BACKEND)
    if int(parameters.get(_iteration_parameter(BACKEND), 0)) != ITERATIONS:
        raise RuntimeError("configs/p1.toml no longer specifies frozen 700-tree XGBoost")
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for fold_number, (fold_name, train_idx, val_idx) in enumerate(folds):
        postprocess = dict(FOLD_POSTPROCESS[fold_name])
        fold_parameters = dict(parameters)
        fold_parameters[_iteration_parameter(BACKEND)] = ITERATIONS

        # Secondary incumbent and primary comparator A share the frozen model.
        baseline_encoder = TabularEncoder().fit(bundle, train_idx)
        baseline_train = baseline_encoder.transform(bundle, train_idx)
        baseline_validation = baseline_encoder.transform(bundle, val_idx)
        # Training labels are permitted model-fit inputs.  Only outer
        # validation label values are embargoed until A and B are complete.
        fit_target = train.iloc[train_idx]["label"].to_numpy(dtype=np.int8)
        baseline_model = _fit_model(
            BACKEND,
            fold_parameters,
            config.seed + fold_number,
            _threads(config),
            baseline_train,
            fit_target,
        )
        secondary_probability = baseline_model.predict_proba(baseline_validation)[:, 1]
        validation_frame = train.iloc[val_idx].drop(columns=["label", "anomaly_type"]).copy()
        plateau = detect_plateaus(validation_frame).to_numpy(dtype=bool)
        spike = detect_singleton_spikes(validation_frame).to_numpy(dtype=bool)
        secondary_prediction = apply_postprocess(
            validation_frame,
            secondary_probability,
            plateau,
            spike,
            postprocess,
        )
        reference_view = reference_oof.loc[:, [*KEY_COLUMNS, "fold", "probability", "prediction"]]
        reference = _fold_reference(reference_view, fold_name, validation_frame)
        maximum_error = float(
            np.max(
                np.abs(secondary_probability - reference["probability"].to_numpy(dtype=np.float64))
            )
        )
        mismatches = int(
            (secondary_prediction != reference["prediction"].to_numpy(dtype=np.int8)).sum()
        )
        if maximum_error > PROBABILITY_TOLERANCE or mismatches:
            raise RuntimeError(f"{fold_name} frozen OOF reproduction failed before A/B comparison")

        station = validation_frame["station"].astype("string")
        gors = station.eq(GORS_STATION).to_numpy(dtype=bool)
        fallback_codes = depth_fallback_codes(
            baseline_encoder,
            validation_frame["station"],
            validation_frame["layer"],
        )
        a_features = apply_depth_missing_counterfactual(
            baseline_validation,
            bundle.feature_columns,
            gors,
            fallback_codes,
        )
        a_probability = baseline_model.predict_proba(a_features)[:, 1]
        a_prediction = apply_postprocess(
            validation_frame,
            a_probability,
            plateau,
            spike,
            postprocess,
        )

        candidate = encode_gors_depth_invariant_fold(
            train,
            bundle,
            train_idx,
            val_idx,
        )
        non_g_train = train.iloc[train_idx]["station"].astype("string").ne(GORS_STATION)
        non_g_validation = station.ne(GORS_STATION)
        if not np.array_equal(
            candidate.train_features[non_g_train.to_numpy()],
            baseline_train[non_g_train.to_numpy()],
            equal_nan=True,
        ):
            raise RuntimeError(f"{fold_name} non-G fold-train encoded inputs changed")
        if not np.array_equal(
            candidate.validation_features[non_g_validation.to_numpy()],
            baseline_validation[non_g_validation.to_numpy()],
            equal_nan=True,
        ):
            raise RuntimeError(f"{fold_name} non-G validation encoded inputs changed")
        candidate_model = _fit_model(
            BACKEND,
            fold_parameters,
            config.seed + fold_number,
            _threads(config),
            candidate.train_features,
            fit_target,
        )
        b_probability = candidate_model.predict_proba(candidate.validation_features)[:, 1]
        b_prediction = apply_postprocess(
            validation_frame,
            b_probability,
            plateau,
            spike,
            postprocess,
        )

        output = validation_frame.loc[:, ["station", "year", "layer", "time"]].copy()
        output["fold"] = fold_name
        output["secondary_probability"] = secondary_probability.astype(np.float32)
        output["secondary_prediction"] = secondary_prediction
        output["a_probability"] = a_probability.astype(np.float32)
        output["a_prediction"] = a_prediction
        output["b_probability"] = b_probability.astype(np.float32)
        output["b_prediction"] = b_prediction
        output["gors"] = gors
        # Labels are intentionally absent from this intermediate object.
        parts.append(output)
        audit_rows.append(
            {
                "fold": fold_name,
                "iterations": ITERATIONS,
                "postprocess": postprocess,
                "frozen_probability_max_abs_error": maximum_error,
                "frozen_prediction_mismatch_rows": mismatches,
                "gors_validation_rows": int(gors.sum()),
                "non_g_train_encoded_inputs_bitwise_equal": True,
                "non_g_validation_encoded_inputs_bitwise_equal": True,
            }
        )

    return pd.concat(parts, ignore_index=True), {
        "folds": audit_rows,
        "all_predictions_completed_before_outer_label_access": True,
        "inner_reselection": False,
        "iterations": ITERATIONS,
    }


def _relative_increase(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - baseline) / baseline


def _append_exposure_event(
    path: Path,
    *,
    event: str,
    run_id: str,
    preregistration_sha256: str,
    decision: str | None = None,
) -> None:
    """Append and fsync a one-shot exposure event before truth is accessed."""

    if event not in {"outer_evaluated", "closed"}:
        raise ValueError(f"unsupported exposure event: {event}")
    payload: dict[str, Any] = {
        "event": event,
        "experiment_id": "P1_gors_depth_invariance_v1",
        "family_id": "gors_deployment_depth_invariance",
        "run_id": run_id,
        "preregistration_sha256": preregistration_sha256,
        "outer_result_count": 1 if event == "outer_evaluated" else 0,
    }
    if decision is not None:
        payload["decision"] = decision
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _evaluate_completed_predictions(
    prediction_frame: pd.DataFrame,
    train: pd.DataFrame,
    test_shares: Mapping[Any, float],
    *,
    reference_weighted_f1: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach outer truth only after all A/B predictions are fixed, then score."""

    keys = ["station", "year", "layer", "time"]
    labels = train.loc[:, [*keys, "label", "anomaly_type"]]
    complete = prediction_frame.merge(labels, on=keys, how="left", validate="one_to_one")
    if complete["label"].isna().any():
        raise RuntimeError("outer labels did not align after predictions were completed")
    truth = complete["label"].to_numpy(dtype=np.int8)
    a = complete["a_prediction"].to_numpy(dtype=np.int8)
    b = complete["b_prediction"].to_numpy(dtype=np.int8)
    secondary = complete["secondary_prediction"].to_numpy(dtype=np.int8)
    gors = complete["gors"].to_numpy(dtype=bool)
    non_g = ~gors

    a_report = evaluate_predictions(
        truth,
        a,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    b_report = evaluate_predictions(
        truth,
        b,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    secondary_report = evaluate_predictions(
        truth,
        secondary,
        complete,
        group_weights=test_shares,
        anomaly_type=complete["anomaly_type"],
    )
    if abs(a_report.weighted.f1 - A_EXPECTED_WEIGHTED_F1) > A_REPRODUCTION_TOLERANCE:
        raise RuntimeError("newly produced comparator A weighted F1 failed exact reproduction")
    a_gors = binary_counts(truth[gors], a[gors])
    b_gors = binary_counts(truth[gors], b[gors])
    if abs(a_gors.f1 - A_EXPECTED_GORS_F1) > A_REPRODUCTION_TOLERANCE:
        raise RuntimeError("newly produced comparator A G-ORS F1 failed exact reproduction")

    non_g_weights = {
        key: weight for key, weight in test_shares.items() if str(key[0]) != GORS_STATION
    }
    # weighted_group_counts renormalizes the retained target row shares over
    # observed non-G groups; this definition is fixed before evaluation.
    a_non_g = weighted_group_counts(truth[non_g], a[non_g], complete.loc[non_g], non_g_weights)
    b_non_g = weighted_group_counts(truth[non_g], b[non_g], complete.loc[non_g], non_g_weights)

    fold_deltas = []
    for fold_name, part in complete.groupby("fold", sort=False, observed=True):
        fold_truth = part["label"].to_numpy(dtype=np.int8)
        fold_weights = {
            key: value
            for key, value in test_shares.items()
            if key in set(part.loc[:, ["station", "layer"]].itertuples(index=False, name=None))
        }
        fold_delta = (
            weighted_group_counts(fold_truth, part["b_prediction"], part, fold_weights).f1
            - weighted_group_counts(fold_truth, part["a_prediction"], part, fold_weights).f1
        )
        fold_gors_support = int(part.loc[part["gors"].astype(bool), "label"].sum())
        fold_deltas.append(
            {
                "fold": str(fold_name),
                "b_minus_a_test_share_weighted_f1": float(fold_delta),
                "non_degrading_at_tolerance_1e_12": bool(fold_delta >= -1.0e-12),
                "gors_positive_rows": fold_gors_support,
                "gors_f1_is_informative": fold_gors_support > 0,
            }
        )
    q4 = next(row for row in fold_deltas if row["fold"] == "2025_q4")
    if q4["gors_positive_rows"] != 0:
        raise RuntimeError("frozen Q4 G-ORS positive support is no longer zero")

    group_drops: list[dict[str, Any]] = []
    for key, part in complete.loc[non_g].groupby(["station", "layer"], sort=True, observed=True):
        group_truth = part["label"].to_numpy(dtype=np.int8)
        delta = (
            binary_counts(group_truth, part["b_prediction"]).f1
            - binary_counts(group_truth, part["a_prediction"]).f1
        )
        group_drops.append(
            {"station": str(key[0]), "layer": int(key[1]), "b_minus_a_f1": float(delta)}
        )
    worst_non_g_drop = min((row["b_minus_a_f1"] for row in group_drops), default=0.0)

    bootstrap = paired_block_bootstrap(
        truth,
        b,
        a,
        complete,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        normal_day_timezone="Asia/Seoul",
    )
    normal_fp = normal_station_layer_day_fp(truth, b, a, complete)
    candidate_fp_day = float(
        normal_fp["candidate"]["false_positive_rows_per_normal_station_layer_day"]
    )
    baseline_fp_day = float(
        normal_fp["baseline"]["false_positive_rows_per_normal_station_layer_day"]
    )
    fp_relative = _relative_increase(candidate_fp_day, baseline_fp_day)
    fp_pass = (
        candidate_fp_day == 0
        if baseline_fp_day == 0
        else bool(
            fp_relative is not None
            and fp_relative < EXACT_GATES["normal_fp_day_relative_increase_lt"]
        )
    )
    weighted_delta = b_report.weighted.f1 - a_report.weighted.f1
    gates = {
        "b_minus_a_test_share_weighted_f1_delta": float(weighted_delta),
        "test_share_weighted_f1_delta_pass": bool(
            weighted_delta >= EXACT_GATES["test_share_weighted_f1_delta_min"] - 1.0e-12
        ),
        "paired_bootstrap_ci90_lower": float(bootstrap["difference_ci90"][0]),
        "paired_bootstrap_pass": bool(
            bootstrap["difference_ci90"][0] > EXACT_GATES["paired_bootstrap_90pct_lower_bound_gt"]
        ),
        "b_minus_a_gors_f1_delta": float(b_gors.f1 - a_gors.f1),
        "gors_f1_delta_pass": bool(
            b_gors.f1 - a_gors.f1 >= EXACT_GATES["gors_group_f1_delta_min"] - 1.0e-12
        ),
        "b_minus_a_non_g_weighted_f1_delta": float(b_non_g.f1 - a_non_g.f1),
        "non_g_weighted_f1_pass": bool(
            b_non_g.f1 - a_non_g.f1
            >= EXACT_GATES["non_g_test_share_weighted_f1_delta_min"] - 1.0e-12
        ),
        "worst_non_g_station_layer_f1_delta": float(worst_non_g_drop),
        "non_g_station_layer_drop_pass": bool(
            worst_non_g_drop >= -EXACT_GATES["non_g_station_layer_f1_drop_max"] - 1.0e-12
        ),
        "folds_non_degrading": int(
            sum(row["non_degrading_at_tolerance_1e_12"] for row in fold_deltas)
        ),
        "fold_non_degradation_pass": bool(
            sum(row["non_degrading_at_tolerance_1e_12"] for row in fold_deltas)
            >= EXACT_GATES["folds_non_degrading_min"]
        ),
        "normal_fp_day_relative_increase": fp_relative,
        "normal_fp_day_pass": fp_pass,
        "secondary_b_minus_natural_weighted_f1_delta": float(
            b_report.weighted.f1 - reference_weighted_f1
        ),
        "secondary_incumbent_safety_pass": bool(
            b_report.weighted.f1 - reference_weighted_f1 >= -0.001 - 1.0e-12
        ),
    }
    primary_names = (
        "test_share_weighted_f1_delta_pass",
        "paired_bootstrap_pass",
        "gors_f1_delta_pass",
        "non_g_weighted_f1_pass",
        "non_g_station_layer_drop_pass",
        "fold_non_degradation_pass",
        "normal_fp_day_pass",
    )
    gates["all_primary_gates_pass"] = bool(all(gates[name] for name in primary_names))
    gates["promotion_eligible"] = bool(
        gates["all_primary_gates_pass"] and gates["secondary_incumbent_safety_pass"]
    )

    metrics = {
        "primary_comparison": "B_minus_A",
        "a_deployment_matched": a_report.to_dict(),
        "b_symmetric_mask": b_report.to_dict(),
        "secondary_natural_depth": secondary_report.to_dict(),
        "gors": {
            "a": a_gors.to_dict(),
            "b": b_gors.to_dict(),
            "q4_positive_support_is_zero": True,
        },
        "non_g_test_shares_renormalized": True,
        "non_g": {"a": a_non_g.to_dict(), "b": b_non_g.to_dict()},
        "folds": fold_deltas,
        "non_g_station_layer": group_drops,
        "paired_block_bootstrap": {
            **bootstrap,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "block_definition": "positive event plus normal station-layer KST-day",
        },
        "normal_station_layer_day_fp": {
            **normal_fp,
            "zero_baseline_rule": "0 if B is also 0, else +Infinity and fail",
        },
        "gates": gates,
        "outer_is_independent_holdout": False,
    }
    output = complete.loc[
        :,
        [
            *keys,
            "fold",
            "label",
            "secondary_probability",
            "secondary_prediction",
            "a_probability",
            "a_prediction",
            "b_probability",
            "b_prediction",
        ],
    ].copy()
    return output, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve(strict=True)
    preregistration = args.preregistration.resolve(strict=True)
    ledger = args.ledger.resolve(strict=True)
    reference_run = args.reference_run.resolve(strict=True)
    deployment_stress = args.deployment_stress.resolve(strict=True)
    data_env = os.environ.get("P1_DATA_DIR")
    override = args.data_dir or (Path(data_env) if data_env else None)
    if override is None:
        raise OSError("P1_DATA_DIR or --data-dir is required")
    config = load_config(config_path, env={"P1_DATA_DIR": str(override)})
    if config.mode != "offline" or config.features.mode != "offline":
        raise RuntimeError("G-ORS depth invariance is pinned to offline mode")
    data_dir = resolve_data_dir(config, override)
    paths = _input_paths(
        config_path=config_path,
        data_dir=data_dir,
        reference_run=reference_run,
        deployment_stress=deployment_stress,
    )
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != REFERENCE_HASHES:
        raise RuntimeError("observed input hashes differ from the preregistered contract")
    receipt = validate_gors_depth_preregistration_files(
        preregistration,
        ledger_path=ledger,
        observed_hashes=hashes,
        require_outer_authorized=True,
    )
    # The receipt and its canonical preregistration SHA enter the manifest
    # before source rows or labels are loaded and before predictions begin.
    recorder = RunRecorder(
        "gors_depth_invariance_one_shot",
        {"receipt": receipt, "statistical_contract": EXACT_GATES},
        root=(
            config.paths.artifacts_dir
            if config.paths.artifacts_dir.is_absolute()
            else PROJECT_ROOT / config.paths.artifacts_dir
        ),
        seed=config.seed,
    )
    recorder.record_json("preregistration_receipt.json", receipt)
    recorder.add_inputs(**paths, preregistration=preregistration, ledger=ledger)
    try:
        train, test = load_train_test(data_dir, audit=True, strict=True)
        bundle = load_or_build_features(train, config, kind="train", use_cache=True)
        reference_oof = pd.read_parquet(
            paths["oof"],
            columns=[*KEY_COLUMNS, "fold", "probability", "prediction"],
        )
        predictions, execution_audit = _arm_predictions_without_truth(
            train,
            bundle,
            config,
            reference_oof,
        )
        if not execution_audit["all_predictions_completed_before_outer_label_access"]:
            raise RuntimeError("prediction completion assertion failed")
        # Persist exposure before any outer truth-derived artifact or label is
        # opened.  A crash from this point permanently blocks a rerun.
        _append_exposure_event(
            ledger,
            event="outer_evaluated",
            run_id=recorder.run_id,
            preregistration_sha256=receipt["preregistration_sha256"],
        )
        # Only after both arms are complete and exposure is durable do we open
        # the prior truth-derived A aggregate and current outer labels.
        expected_a = _assert_existing_stress(deployment_stress)
        execution_audit["existing_a_exact_postprediction_check"] = expected_a
        oof, metrics = _evaluate_completed_predictions(
            predictions,
            train,
            group_row_shares(test),
            reference_weighted_f1=0.8133155525620019,
        )
        output = recorder.path / "oof.parquet"
        oof.to_parquet(output, index=False, compression="zstd")
        recorder.record_file(output)
        recorder.record_json("execution_audit.json", execution_audit)
        recorder.record_json("metrics.json", metrics)
        recorder.finish(
            status="complete",
            preregistration_sha256=receipt["preregistration_sha256"],
            all_primary_gates_pass=metrics["gates"]["all_primary_gates_pass"],
            promotion_eligible=metrics["gates"]["promotion_eligible"],
            competition_upload=False,
            commit=False,
            push=False,
        )
        _append_exposure_event(
            ledger,
            event="closed",
            run_id=recorder.run_id,
            preregistration_sha256=receipt["preregistration_sha256"],
            decision=("go" if metrics["gates"]["promotion_eligible"] else "no_go"),
        )
    except Exception as exc:
        recorder.finish(
            status="failed",
            preregistration_sha256=receipt["preregistration_sha256"],
            error_type=type(exc).__name__,
            error=str(exc),
            competition_upload=False,
            commit=False,
            push=False,
        )
        raise
    print(
        json.dumps(
            {
                "run_id": recorder.run_id,
                "run_path": str(recorder.path.resolve()),
                "all_primary_gates_pass": metrics["gates"]["all_primary_gates_pass"],
                "promotion_eligible": metrics["gates"]["promotion_eligible"],
                "uploaded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

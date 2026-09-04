"""Run the canonical append-only P3 meaningful learning-curve generation.

The run refits the exact current CatBoost/router/0.20-shrink comparator at every
chronological training prefix.  Three preregistered structural heads reuse those same
component fits.  Anonymous-test values are opened only after a curve-qualified gate;
the runner never uploads or mutates the frozen/current submission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost.utils import get_gpu_device_count

from ocean_goal.meaningful_score import evaluate_learning_curve, load_contract
from p3_wave.corrected_repeated_forward import CorrectedFold, build_corrected_repeated_forward_folds
from p3_wave.meaningful_learning_curve import (
    HYPOTHESES,
    PREFIX_FRACTIONS,
    central_evidence,
    chronological_prefix_ids,
    evaluate_hypothesis_gate,
    evaluate_point,
    hypothesis_predictions,
    next_structural_generation,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.persistence_shrink import LongLeadPersistenceShrink, apply_long_lead_persistence_shrink
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.submission import build_submission, validate_submission
from p3_wave.validation import expand_leads, rmse

_BASE_PATH = Path(__file__).with_name("run_p3_corrected_repeated_forward_catboost_v1.py")
_BASE_SPEC = importlib.util.spec_from_file_location("p3_curve_base_helpers", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("failed to load pinned P3 fit helpers")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)

EXPECTED_CONFIG_SHA256 = "bd8cd670827c314dcb495879348015d9b1fbf580288540f421afee66de3ae81e"
EXPECTED_CONFIG_DEEP_SHA256 = "57a5beaa4321756031fbc6b4d033c62819c8d5fe9a6b8095677997c548b177c0"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_meaningful_learning_curve_v1.json"
CANONICAL_GOAL_RELATIVE = "configs/goals/meaningful_score_maximization_v2.json"
CANONICAL_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1"
CANONICAL_LOCK_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1.ATTEMPT_LOCK.json"
REFERENCE_PATHS = {
    "corrected_v2_metrics": "artifacts/p3_corrected_repeated_forward_catboost_v2/metrics.json",
    "corrected_v2_oof": "artifacts/p3_corrected_repeated_forward_catboost_v2/oof.parquet",
    "fixed_long_shrink_v4_metrics": "artifacts/p3_corrected_fixed_long_shrink_v4/metrics.json",
    "meaningful_improvement_audit": (
        "artifacts/full_improvement_cycle_20260822/meaningful_improvement_audit.json"
    ),
}


def _now() -> str:
    return base._now()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_paths(root: Path) -> dict[str, Path]:
    workspace = root.resolve(strict=True)
    return {
        "config": workspace / CANONICAL_CONFIG_RELATIVE,
        "goal": workspace / CANONICAL_GOAL_RELATIVE,
        "cache": workspace / CANONICAL_CACHE_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
    }


def authorize_entry(
    *,
    root: Path,
    requested_config: Path,
    requested_cache: Path,
    requested_output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """First action: authorize exact path, byte SHA, deep JSON SHA, and immutable pins."""

    paths = _canonical_paths(root)
    requested = {
        "config": requested_config.resolve(strict=True),
        "cache": requested_cache.resolve(strict=True),
        "output": requested_output.resolve(strict=False),
    }
    for name in ("config", "cache", "output"):
        if requested[name] != paths[name].resolve(strict=name != "output"):
            raise PermissionError(f"non-canonical {name} path is forbidden")
    if paths["output"].exists():
        raise FileExistsError("canonical append-only output already exists")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if _deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config fails compiled deep-JSON equality")
    expected_paths = {
        "config": CANONICAL_CONFIG_RELATIVE,
        "goal_contract": CANONICAL_GOAL_RELATIVE,
        "cache": CANONICAL_CACHE_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
    }
    if config.get("canonical_paths") != expected_paths:
        raise PermissionError("canonical path fields differ")
    if config.get("experiment_id") != "p3_meaningful_learning_curve_v1":
        raise PermissionError("experiment identity differs")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise PermissionError("prefix curve differs")
    if tuple(item["id"] for item in config["hypotheses"]) != HYPOTHESES:
        raise PermissionError("hypothesis order differs")
    if config["model"]["seed_replicates"] != [20260816, 20260817, 20260818]:
        raise PermissionError("three fixed seeds differ")
    if config["shrink"] != {
        "active_leads": [12, 18, 24],
        "persistence_weight": 0.2,
        "applied_identically_to_comparator_and_all_three_hypotheses": True,
    }:
        raise PermissionError("fixed comparator postprocess differs")
    if not all(config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")

    implementation_paths = {
        "base_runner": root / "scripts/run_p3_corrected_repeated_forward_catboost_v1.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score.py",
    }
    for name, path in implementation_paths.items():
        if base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    for name, relative in REFERENCE_PATHS.items():
        if base.sha256_file(root / relative) != config["reference_evidence_sha256"][name]:
            raise PermissionError(f"reference evidence SHA differs: {name}")
    return config, paths


def _prefix_id_sha(ids: np.ndarray) -> str:
    canonical = np.asarray(ids, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _preflight(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    pinned = base._resolved_input_paths(root=root, data_dir=data_dir, cache_dir=paths["cache"])
    snapshot = base._verify_input_hashes(pinned, config["expected_sha256"])
    if get_gpu_device_count() != 1:
        raise RuntimeError("canonical curve run requires exactly one visible GPU")
    features = pd.read_parquet(paths["cache"] / "train_features.parquet")
    anchors = pd.read_parquet(paths["cache"] / "train_anchors.parquet")
    if len(features) != 24_360 or len(anchors) != 24_360:
        raise ValueError("training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("training feature/anchor keys differ")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    if len(feature_columns) != config["features"]["expected_feature_count"]:
        raise ValueError("feature count differs")
    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=config["validation"]["gap_hours"],
        footprint_hours=config["validation"]["footprint_hours"],
    )
    if len(selected) != 181 or split_audit["validation_row_count"] != 1_086:
        raise ValueError("corrected validation surface differs")
    prefix_audit: dict[str, Any] = {}
    lookup = anchors.set_index("anchor_id")
    for fraction in PREFIX_FRACTIONS:
        name = f"{int(round(fraction * 100)):03d}"
        prefix_audit[name] = {}
        for fold in folds:
            ids = chronological_prefix_ids(anchors, fold.train_ids, fraction)
            times = pd.to_datetime(lookup.loc[ids, "anchor_time"], utc=True)
            validation_start = pd.Timestamp(fold.validation_start)
            maximum = times.max()
            if maximum >= validation_start - pd.Timedelta(hours=78):
                raise AssertionError("prefix violates the 78-hour embargo")
            prefix_audit[name][fold.name] = {
                "fraction": float(fraction),
                "count": int(len(ids)),
                "full_count": int(len(fold.train_ids)),
                "id_sha256_little_endian_int64": _prefix_id_sha(ids),
                "nested_subset_of_safe_outer_train": bool(np.isin(ids, fold.train_ids).all()),
                "maximum_anchor_before_validation_start_hours": float(
                    (validation_start - maximum).total_seconds() / 3600.0
                ),
            }
    leakage_checks = {
        "station_global_validation_gap_at_least_78h": all(
            value >= 78.0 for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_station_episode_reuse_zero": split_audit["repeated_station_episode_count"] == 0,
        "validation_72h_footprint_overlap_zero": split_audit[
            "context48_plus_target24_footprint_overlap_pairs"
        ]
        == 0,
        "outer_train_validation_episode_overlap_zero": all(
            row["shared_train_validation_station_episode_count"] == 0
            for row in split_audit["folds"].values()
        ),
        "outer_train_validation_gap_at_least_78h": all(
            row["minimum_train_validation_anchor_gap_hours"] >= 78.0
            for row in split_audit["folds"].values()
        ),
        "all_prefixes_nested_in_safe_outer_train": all(
            row["nested_subset_of_safe_outer_train"]
            for folds_at_fraction in prefix_audit.values()
            for row in folds_at_fraction.values()
        ),
    }
    if not all(leakage_checks.values()):
        raise AssertionError("preflight leakage checks failed")
    return {
        "pinned_paths": pinned,
        "input_snapshot": snapshot,
        "features": features,
        "anchors": anchors,
        "feature_columns": feature_columns,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "prefix_audit": prefix_audit,
        "leakage_checks": leakage_checks,
    }


def _prefix_fold(fold: CorrectedFold, anchors: pd.DataFrame, fraction: float) -> CorrectedFold:
    return CorrectedFold(
        name=fold.name,
        train_ids=chronological_prefix_ids(anchors, fold.train_ids, fraction),
        validation_ids=fold.validation_ids.copy(),
        validation_start=fold.validation_start,
        validation_end=fold.validation_end,
    )


def _seed_mean_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) != 3:
        raise ValueError("exactly three seed frames are required")
    keys = ["fold", "anchor_id", "station", "lead_h"]
    invariant = ["target_hs", "current_hs", "persistence"]
    prediction = ["incumbent_prediction", *HYPOTHESES]
    ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in frames]
    reference = ordered[0][keys + invariant]
    for frame in ordered[1:]:
        if not frame[keys].equals(reference[keys]):
            raise ValueError("seed OOF keys differ")
        if not np.array_equal(
            frame[invariant].to_numpy(float), reference[invariant].to_numpy(float)
        ):
            raise ValueError("seed OOF truth/current values differ")
    result = reference.copy()
    for column in prediction:
        result[column] = np.mean(
            np.column_stack([frame[column].to_numpy(float) for frame in ordered]), axis=1
        )
    if not np.isfinite(result[prediction].to_numpy(float)).all():
        raise ValueError("seed-mean predictions are non-finite")
    return result


def _run_curve(
    *, root: Path, preflight: dict[str, Any], config: dict[str, Any], stage: Path
) -> tuple[
    pd.DataFrame,
    dict[str, dict[float, dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    features: pd.DataFrame = preflight["features"]
    anchors: pd.DataFrame = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    selected: pd.DataFrame = preflight["selected"]
    feature_columns: list[str] = preflight["feature_columns"]
    all_prefix_frames: list[pd.DataFrame] = []
    points: dict[str, dict[float, dict[str, Any]]] = {name: {} for name in HYPOTHESES}
    receipts: list[dict[str, Any]] = []
    exact_comparator: dict[str, Any] = {
        "checked": False,
        "exact_prediction_equal": False,
        "maximum_absolute_difference_m": None,
    }
    shrink = LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24))

    for fraction in PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        seed_frames: list[pd.DataFrame] = []
        for seed in config["model"]["seed_replicates"]:
            seed_component_frames: list[pd.DataFrame] = []
            seed_config = copy.deepcopy(config)
            seed_config["model"]["fold_seeds"] = [int(seed)] * 3
            seed_config["validation"]["bootstrap_replicates"] = 1
            for fold_number, fold in enumerate(folds):
                current_fold = _prefix_fold(fold, anchors, fraction)
                print(
                    json.dumps(
                        {
                            "phase": "fit_prefix_cell",
                            "prefix": fraction,
                            "seed": seed,
                            "fold": fold.name,
                            "train_cases": len(current_fold.train_ids),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                current, receipt = base._fit_fold_components(
                    fold=current_fold,
                    fold_number=fold_number,
                    features=features,
                    anchors=anchors,
                    feature_columns=feature_columns,
                    config=seed_config,
                    model_dir=stage / "models" / prefix_tag / f"seed_{seed}" / "folds",
                )
                receipt.update(
                    {
                        "prefix_fraction": float(fraction),
                        "seed_replicate": int(seed),
                        "cell_elapsed_seconds": float(time.perf_counter() - started),
                    }
                )
                receipts.append(receipt)
                seed_component_frames.append(current)
            component_oof = pd.concat(seed_component_frames, ignore_index=True)
            evaluated, _, _ = base._evaluate_fixed_structure(
                component_oof=component_oof,
                train_features=features,
                anchors=anchors,
                fold_order=tuple(fold.name for fold in folds),
                config=seed_config,
                split_audit=preflight["split_audit"],
                expected_validation_ids=selected["anchor_id"].to_numpy(dtype=np.int64),
            )
            frame = evaluated[
                [
                    "fold",
                    "anchor_id",
                    "station",
                    "lead_h",
                    "target_hs",
                    "current_hs",
                    "persistence",
                    "single_prediction",
                    "multi_prediction",
                    "equal_prediction",
                    "final_prediction",
                ]
            ].copy()
            frame["incumbent_prediction"] = frame["final_prediction"].to_numpy(float)
            for name, raw in hypothesis_predictions(frame).items():
                frame[name] = apply_long_lead_persistence_shrink(
                    raw,
                    frame["persistence"].to_numpy(float),
                    frame["lead_h"].to_numpy(int),
                    config=shrink,
                )
            seed_frames.append(
                frame[
                    [
                        "fold",
                        "anchor_id",
                        "station",
                        "lead_h",
                        "target_hs",
                        "current_hs",
                        "persistence",
                        "incumbent_prediction",
                        *HYPOTHESES,
                    ]
                ]
            )
        mean_frame = _seed_mean_frame(seed_frames)
        mean_frame["prefix_fraction"] = float(fraction)
        all_prefix_frames.append(mean_frame)
        for index, hypothesis in enumerate(HYPOTHESES):
            points[hypothesis][fraction] = evaluate_point(
                mean_frame,
                candidate_column=hypothesis,
                bootstrap_replicates=config["validation"]["bootstrap_replicates"],
                bootstrap_seed=(
                    int(config["validation"]["bootstrap_seed"])
                    + 1_000 * index
                    + int(round(fraction * 100))
                ),
            )
            points[hypothesis][fraction]["incumbent_seed_metrics"] = [
                float(rmse(frame["target_hs"], frame["incumbent_prediction"]))
                for frame in seed_frames
            ]
            points[hypothesis][fraction]["challenger_seed_metrics"] = [
                float(rmse(frame["target_hs"], frame[hypothesis])) for frame in seed_frames
            ]
        if fraction == 1.0:
            fold_order = tuple(fold.name for fold in folds)
            historical_rebuild = pd.concat(
                [
                    seed_frames[index].loc[seed_frames[index]["fold"].astype(str).eq(fold_name)]
                    for index, fold_name in enumerate(fold_order)
                ],
                ignore_index=True,
            )
            historical = pd.read_parquet(
                root / "artifacts/p3_corrected_repeated_forward_catboost_v2/oof.parquet"
            )
            keys = ["fold", "anchor_id", "station", "lead_h"]
            left = historical_rebuild.sort_values(keys).reset_index(drop=True)
            right = historical.sort_values(keys).reset_index(drop=True)
            if not left[keys].equals(right[keys]):
                raise ValueError("full-prefix refit keys differ from frozen corrected OOF")
            difference = np.abs(
                left["incumbent_prediction"].to_numpy(float)
                - right["final_prediction"].to_numpy(float)
            )
            exact_comparator = {
                "checked": True,
                "exact_prediction_equal": bool(
                    np.array_equal(difference, np.zeros_like(difference))
                ),
                "maximum_absolute_difference_m": float(np.max(difference)),
            }
        print(
            json.dumps(
                {
                    "phase": "prefix_complete",
                    "prefix": fraction,
                    "full_deltas_so_far": {
                        name: points[name][fraction]["delta_candidate_minus_incumbent_m"]
                        for name in HYPOTHESES
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.concat(all_prefix_frames, ignore_index=True), points, receipts, exact_comparator


def _protected_roots(root: Path, data_dir: Path, cache: Path) -> tuple[Path, ...]:
    return (data_dir, cache, root / "submissions", root / "output", root / "데이터셋 원본")


def _write_exclusive_submission(frame: pd.DataFrame, test_index: pd.DataFrame, path: Path) -> None:
    validate_submission(frame, test_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", mode="x")


def _fit_full_candidate(
    *,
    root: Path,
    data_dir: Path,
    cache: Path,
    stage: Path,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: list[str],
    hypothesis: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    access = {
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    seed = int(config["model"]["full_train_seed"])
    train_ids = anchors["anchor_id"].to_numpy(np.int64)
    x_train, y_train, train_meta = expand_leads(features, anchors, train_ids, feature_columns)
    single = base._single_model(config, seed)
    single.fit(
        base._cat_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    multi_train = feature_lookup.loc[train_ids, ["station", *feature_columns]].reset_index(
        drop=True
    )
    multi_train["station"] = multi_train["station"].astype(str)
    multi = base._multi_model(config, seed)
    multi.fit(
        multi_train,
        base._multi_target(anchors, train_ids),
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
        ),
        cat_features=[0],
        verbose=False,
    )

    test_features = pd.read_parquet(cache / "test_features.parquet")
    access["test_feature_cache_value_reads"] += 1
    test_index = pd.read_csv(data_dir / "test_index.csv")
    access["test_index_value_reads"] += 1
    if list(test_index.columns) != base.KEYS or len(test_index) != 1_200:
        raise ValueError("test_index contract differs")
    validate_submission(
        pd.DataFrame({**{key: test_index[key] for key in base.KEYS}, "hs_pred": 0.0}),
        test_index,
    )
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    test_cases = case_order.merge(
        test_features,
        on=["case_id", "station"],
        how="left",
        validate="one_to_one",
    )
    if len(test_cases) != 200 or test_cases[feature_columns].isna().all(axis=1).any():
        raise ValueError("same-case test feature alignment failed")
    forbidden = {
        column
        for column in test_features.columns
        if column.lower() in {"time", "timestamp", "date", "target_hs", "truth", "label"}
    }
    if forbidden:
        raise ValueError(f"forbidden anonymous-test feature columns: {sorted(forbidden)}")
    source = test_cases.set_index(["case_id", "station"])
    repeated = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    single_x = source.loc[repeated, feature_columns].reset_index(drop=True)
    single_x.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    single_x.insert(0, "station", test_index["station"].astype(str).to_numpy())
    current_rows = source.loc[repeated, "hs_current"].to_numpy(float)
    single_x.insert(2, "current_hs_for_residual", current_rows)
    single_prediction = np.clip(current_rows + single.predict(base._cat_frame(single_x)), 0.0, 30.0)
    multi_x = test_cases[["station", *feature_columns]].copy()
    multi_x["station"] = multi_x["station"].astype(str)
    current_case = test_cases["hs_current"].to_numpy(float)
    multi_prediction = np.clip(
        current_case[:, None] + np.asarray(multi.predict(multi_x), dtype=float), 0.0, 30.0
    ).reshape(-1)
    lead = test_index["lead_h"].to_numpy(int)
    if hypothesis == "single_horizon_residual_head":
        raw = single_prediction
    elif hypothesis == "multi_trajectory_residual_head":
        raw = multi_prediction
    elif hypothesis == "fixed_horizon_splice":
        raw = np.where(np.isin(lead, [3, 6, 9, 12]), multi_prediction, single_prediction)
    else:
        raise ValueError("unknown selected hypothesis")
    final = apply_long_lead_persistence_shrink(
        raw,
        current_rows,
        lead,
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    candidate = build_submission(test_index, final)
    validate_submission(candidate, test_index)
    protected = _protected_roots(root, data_dir, cache)
    candidate_path = safe_new_stage_path(
        stage, config["output"]["candidate_relative_path"], protected_roots=protected
    )
    reproduced_path = safe_new_stage_path(
        stage,
        config["output"]["reproduced_candidate_relative_path"],
        protected_roots=protected,
    )
    _write_exclusive_submission(candidate, test_index, candidate_path)
    _write_exclusive_submission(candidate, test_index, reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise AssertionError("saved full-fit candidate is not byte reproducible")
    reread = pd.read_csv(candidate_path)
    validate_submission(reread, test_index)
    if not reread[base.KEYS].equals(test_index[base.KEYS]):
        raise AssertionError("candidate key/order changed after serialization")
    model_dir = stage / "models" / "full"
    model_dir.mkdir(parents=True, exist_ok=True)
    single_path = model_dir / "single.cbm"
    multi_path = model_dir / "multi.cbm"
    single.save_model(single_path)
    multi.save_model(multi_path)
    candidate_sha = base.sha256_file(candidate_path)
    receipt = {
        "status": "LOCAL_CURVE_QUALIFIED_CANDIDATE_CREATED_NOT_UPLOADED",
        "hypothesis": hypothesis,
        "rows": 1_200,
        "cases": 200,
        "key_order_exact": True,
        "finite": bool(np.isfinite(final).all()),
        "range_0_to_30_m": bool(np.all((final >= 0.0) & (final <= 30.0))),
        "same_case_only": True,
        "byte_reproduced": True,
        "candidate_sha256": candidate_sha,
        "reproduced_candidate_sha256": base.sha256_file(reproduced_path),
        "model_sha256": {
            "single.cbm": base.sha256_file(single_path),
            "multi.cbm": base.sha256_file(multi_path),
        },
        "uploaded": False,
    }
    base._atomic_json(stage / "candidate" / "validation.json", receipt)
    return receipt, access


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(stage).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": base.sha256_file(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _run_after_lock(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    print(
        json.dumps(
            {
                "phase": "preflight_pass",
                "validation_cases": 181,
                "prefixes": list(PREFIX_FRACTIONS),
                "seeds": config["model"]["seed_replicates"],
                "fit_cells": 45,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_meaningful_curve_v1_", dir=tmp_root))
    curve_oof, points, receipts, exact_comparator = _run_curve(
        root=root, preflight=preflight, config=config, stage=stage
    )
    base._atomic_parquet(stage / "oof" / "learning_curve_oof.parquet", curve_oof)
    base._atomic_parquet(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )
    base._atomic_json(stage / "feature_columns.json", {"columns": preflight["feature_columns"]})

    reproducibility_checks = {
        "canonical_config_path_sha_and_deep_json_equal": True,
        "exact_current_model_router_and_0p20_shrink_refit_each_prefix": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": bool(
            exact_comparator["exact_prediction_equal"]
        ),
        "same_prefix_ids_for_comparator_and_all_hypotheses": True,
        "fixed_three_seed_replicates_and_mean_reducer": True,
        "same_metric_clip_and_fixed_shrink_for_all_arms": True,
        "hyperparameter_alpha_shrink_and_weight_search_zero": True,
        "complete_unique_181_case_1086_row_surface_each_prefix": all(
            len(curve_oof.loc[curve_oof["prefix_fraction"].eq(fraction)]) == 1_086
            for fraction in PREFIX_FRACTIONS
        ),
        "all_fold_models_saved_and_hashed": len(receipts) == 45,
    }
    gates = {
        name: evaluate_hypothesis_gate(
            points[name],
            leakage_checks=preflight["leakage_checks"],
            reproducibility_checks=reproducibility_checks,
        )
        for name in HYPOTHESES
    }
    passing = [name for name in HYPOTHESES if gates[name]["passed"]]
    selected_hypothesis = (
        passing[0]
        if passing
        else min(
            HYPOTHESES,
            key=lambda name: points[name][1.0]["delta_candidate_minus_incumbent_m"],
        )
    )
    evidence = central_evidence(
        points[selected_hypothesis],
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence["preregistration"] = {
        "generation_id": config["experiment_id"],
        "config_path": CANONICAL_CONFIG_RELATIVE,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "created_before_first_fit": True,
        "hypothesis_count": len(HYPOTHESES),
        "hypothesis_count_at_most_3": len(HYPOTHESES) <= 3,
        "score_derived_tuning": False,
    }
    evidence["curve_protocol"] = {
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "seed_ids": [int(value) for value in config["model"]["seed_replicates"]],
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": bool(
            exact_comparator["exact_prediction_equal"]
        ),
        "frozen_oof_maximum_absolute_difference_m": exact_comparator[
            "maximum_absolute_difference_m"
        ],
    }
    for point in evidence["points"]:
        fraction = float(point["fraction"])
        point["incumbent_seed_metrics"] = list(
            points[selected_hypothesis][fraction]["incumbent_seed_metrics"]
        )
        point["challenger_seed_metrics"] = list(
            points[selected_hypothesis][fraction]["challenger_seed_metrics"]
        )
    central_decision = evaluate_learning_curve(
        load_contract(root, CANONICAL_GOAL_RELATIVE), evidence
    )
    if bool(central_decision["passed"]) != bool(gates[selected_hypothesis]["passed"]):
        raise AssertionError("local and central meaningful-curve decisions differ")
    base._atomic_json(stage / "learning_curve_evidence.json", evidence)

    access = {
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    candidate_receipt = None
    if passing:
        candidate_receipt, access = _fit_full_candidate(
            root=root,
            data_dir=data_dir,
            cache=paths["cache"],
            stage=stage,
            features=preflight["features"],
            anchors=preflight["anchors"],
            feature_columns=preflight["feature_columns"],
            hypothesis=selected_hypothesis,
            config=config,
        )
    diagnosis = None if passing else next_structural_generation(points[selected_hypothesis])
    if diagnosis is not None and diagnosis["count"] != 1:
        raise AssertionError("no-pass diagnosis must contain exactly one next generation")

    input_after = base._verify_input_hashes(preflight["pinned_paths"], config["expected_sha256"])
    if input_after != preflight["input_snapshot"]:
        raise RuntimeError("source/cache/current/frozen inputs changed during run")
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": (
            "CURVE_QUALIFIED_LOCAL_CANDIDATE_CREATED_NOT_UPLOADED"
            if passing
            else "NO_HYPOTHESIS_CURVE_QUALIFIED_STOPPED_BEFORE_TEST_READS"
        ),
        "interpretation": (
            "Corrected same-surface research evidence; not an official hidden score and not "
            "upload authorization. The three-seed mean is an evaluation reducer, while each "
            "cell refits the exact current model/router/0.20-shrink structure."
        ),
        "one_shot_attempt": attempt,
        "hypotheses_preregistered": list(HYPOTHESES),
        "selected_for_central_evidence": selected_hypothesis,
        "passing_hypotheses": passing,
        "points_by_hypothesis": {
            name: {str(fraction): points[name][fraction] for fraction in PREFIX_FRACTIONS}
            for name in HYPOTHESES
        },
        "gates_by_hypothesis": gates,
        "central_goal_evaluator": central_decision,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "exact_full_prefix_comparator_reproduction": exact_comparator,
        "training_receipts": receipts,
        "candidate_validation": candidate_receipt,
        "next_structural_generation": diagnosis,
        "access_counters": access,
        "invariants": {
            "append_only": True,
            "model_or_hyperparameter_search_run": False,
            "shrink_alpha_or_weight_micro_tuning_run": False,
            "test_target_or_hidden_label_reads": 0,
            "absolute_test_timestamp_recovered": False,
            "current_or_frozen_submission_mutated": False,
            "official_submission_uploads": 0,
            "team_wide_daily_upload_limit_assumed": False,
            "source_cache_current_frozen_sha_unchanged": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    base._atomic_json(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "hypotheses": [
            {
                "id": name,
                "curve_qualified": bool(gates[name]["passed"]),
                "full_delta_m": float(points[name][1.0]["delta_candidate_minus_incumbent_m"]),
                "promotion_status": (
                    "local_candidate_not_uploaded" if name in passing else "research_only"
                ),
            }
            for name in HYPOTHESES
        ],
        "candidate_created": candidate_receipt is not None,
        "candidate_uploaded": False,
        "current_frozen_sha256": config["expected_sha256"]["current/ready_submission.csv"],
        "current_frozen_unchanged": True,
        "next_structural_generation_count": 0 if diagnosis is None else 1,
    }
    base._atomic_json(stage / "registry.json", registry)
    implementation_paths = {
        "config": paths["config"],
        "runner": Path(__file__).resolve(),
        "base_runner": _BASE_PATH,
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score.py",
        "tests": root / "tests/test_p3_meaningful_learning_curve_v1.py",
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "append_only_generation": True,
        "canonical_contract": {
            "config_path": CANONICAL_CONFIG_RELATIVE,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
            "config_full_deep_equality": True,
            "cache_path": CANONICAL_CACHE_RELATIVE,
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
        },
        "implementation_sha256": {
            name: base.sha256_file(path) for name, path in implementation_paths.items()
        },
        "git": base._git_state(root),
        "input_sha256_before": preflight["input_snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "curve_qualified": bool(passing),
        "candidate_created": candidate_receipt is not None,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    if paths["output"].exists():
        raise FileExistsError("canonical output appeared before atomic move")
    stage.replace(paths["output"])
    result = {
        "status": metrics["status"],
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": base.sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": base.sha256_file(paths["output"] / "oof" / "learning_curve_oof.parquet"),
        "learning_curve_evidence_sha256": base.sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": base.sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": (
            candidate_receipt["candidate_sha256"] if candidate_receipt is not None else None
        ),
        "passing_hypotheses": passing,
        "selected_for_central_evidence": selected_hypothesis,
        "central_decision": central_decision["decision"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "official_upload_count": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        requested_config=paths["config"],
        requested_cache=paths["cache"],
        requested_output=paths["output"],
    )
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "validation_cases": int(len(preflight["selected"])),
        "validation_rows": int(preflight["split_audit"]["validation_row_count"]),
        "prefix_fit_cells": 45,
        "hypotheses": list(HYPOTHESES),
        "leakage_checks": preflight["leakage_checks"],
        "output_absent": not paths["output"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
        "test_value_reads": 0,
        "upload_count": 0,
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        requested_config=paths["config"],
        requested_cache=paths["cache"],
        requested_output=paths["output"],
    )
    attempt = acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        attempt=attempt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    result = (
        check_only(root=root, data_dir=data_dir)
        if args.check_only
        else run_experiment(root=root, data_dir=data_dir)
    )
    if args.check_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

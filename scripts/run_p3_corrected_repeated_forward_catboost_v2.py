"""Canonical one-shot P3 corrected repeated-forward CatBoost run.

This wrapper fail-closes before data access: exact canonical config path, byte SHA, full deep
equality, implementation pins, cache/output identity, and a persistent O_EXCL attempt lock are
mandatory.  It reuses only the already-tested v1 fit/evaluation helpers; v1 itself is archived as
INVALID_ABORTED and is not eligible evidence.
"""

from __future__ import annotations

import argparse
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

from p3_wave.corrected_repeated_forward import (
    CorrectedFold,
    build_corrected_repeated_forward_folds,
)
from p3_wave.models import compact_feature_columns
from p3_wave.one_shot_guard import (
    acquire_persistent_attempt_lock,
    authorize_canonical_contract,
    safe_new_stage_path,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.submission import validate_submission

_BASE_PATH = Path(__file__).with_name("run_p3_corrected_repeated_forward_catboost_v1.py")
_BASE_SPEC = importlib.util.spec_from_file_location("p3_corrected_v1_base_helpers", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("failed to load pinned v1 helper module")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)

EXPECTED_CONFIG_SHA256 = "e5c2eff7bc9fcd44759d0bc30d965c86eca10c038807f186b1379131aed9b169"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_corrected_repeated_forward_catboost_v2.json"
CANONICAL_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_corrected_repeated_forward_catboost_v2"
CANONICAL_LOCK_RELATIVE = "artifacts/p3_corrected_repeated_forward_catboost_v2.ATTEMPT_LOCK.json"

EXPECTED_CONFIG_JSON = r"""
{
  "experiment_id": "p3_corrected_repeated_forward_catboost_v2",
  "status": "PREREGISTERED_ONE_SHOT_CORRECTED_RESEARCH_EVIDENCE_NOT_UPLOAD_AUTHORIZATION",
  "problem": "P3_wave_forecast",
  "target": "significant_wave_height_hs",
  "metric": "pooled_row_RMSE_m",
  "canonical_paths": {
    "config": "configs/experiments/p3_corrected_repeated_forward_catboost_v2.json",
    "cache": "artifacts/p3/features_all20_v1",
    "output": "artifacts/p3_corrected_repeated_forward_catboost_v2",
    "attempt_lock": "artifacts/p3_corrected_repeated_forward_catboost_v2.ATTEMPT_LOCK.json"
  },
  "implementation_sha256": {
    "base_runner": "4539f3e1778213a12485d204c01de60530178fc1db235496ad38dc16602d3dd5",
    "corrected_split_module": "4b614e2d0fd2259c77d462a69eff27245579d98671ec2a083ce4116278417474",
    "one_shot_guard_module": "9bed88b1f0ec48180ef2a350dc75fd3376dc08b2a903c90fcf00a8dd191f08e5"
  },
  "validation": {
    "windows": [
      ["2024_h2_storm", "2024-07-01", "2024-11-01"],
      ["winter_transition", "2024-11-01", "2025-03-01"],
      ["2025_h1", "2025-03-01", "2025-06-25"]
    ],
    "selection": "station_global_chronological_first_eligible_across_all_windows",
    "gap_hours": 78,
    "context_hours": 48,
    "target_hours": 24,
    "footprint_hours": 72,
    "storm_threshold_hs_m": 1.5,
    "raw_wave_episode_contiguous_minutes": 20,
    "episode_reuse_within_validation": false,
    "same_episode_train_validation": false,
    "bootstrap_replicates": 5000,
    "bootstrap_seed": 20260822
  },
  "features": {
    "surface": "compact_feature_columns_fixed_v1",
    "expected_feature_count": 591,
    "test_feature_semantics": "one row per anonymous case, derived only from that case's 48-hour context"
  },
  "model": {
    "single": {
      "loss_function": "RMSE",
      "iterations": 700,
      "learning_rate": 0.035,
      "depth": 6,
      "l2_leaf_reg": 8.0,
      "random_strength": 0.2,
      "thread_count": 8
    },
    "multi": {
      "loss_function": "MultiRMSE",
      "iterations": 1200,
      "learning_rate": 0.03,
      "depth": 7,
      "l2_leaf_reg": 10.0,
      "random_strength": 0.15,
      "task_type": "GPU",
      "devices": "0",
      "boosting_type": "Plain"
    },
    "fold_seeds": [20260816, 20260817, 20260818],
    "full_train_seed": 20260817,
    "single_weight": 0.5,
    "multi_weight": 0.5,
    "residual_target": "target_hs_minus_current_hs",
    "case_weight": "exp(-0.45*max(current_hs-1.5,0))_mean_normalized",
    "prediction_clip_m": [0.0, 30.0]
  },
  "router": {
    "granularity": "lead_long",
    "name": "smooth_medium",
    "alpha": 10.0,
    "temperature_multiplier": 2.0,
    "strength": 0.5,
    "active_leads": [12, 18, 24],
    "fold_one": "exact_equal_component_no_op",
    "later_folds": "fixed_config_refit_on_completed_corrected_oof_only",
    "hyperparameter_search": false
  },
  "shrink": {
    "active_leads": [12, 18, 24],
    "persistence_weight": 0.2
  },
  "gate": {
    "baseline": "persistence_on_identical_corrected_cases",
    "pooled_rmse_below_persistence": true,
    "paired_case_bootstrap_ci90_upper_below_zero": true,
    "minimum_improved_folds": 2,
    "all_split_and_integrity_contracts": true,
    "on_pass": "full_train_then_same_case_only_200_test_inference",
    "on_fail": "stop_before_test_feature_or_test_index_read"
  },
  "expected_sha256": {
    "source/train_wave.csv": "64b015499785b004eb1df60104c6e24b344b09b80b1bd66681dfff1e5881aa2e",
    "source/train_atmos.csv": "5bf0491a0a781f1084055e40bb4065916c7b6c2f555f38a819199bca3427615f",
    "source/test_context.parquet": "4051ce6af55c320cbaad9a0d3367c654349f7ff3965709b7132d45957d3f1c0c",
    "source/test_index.csv": "004551346ca5be6e3445d8b8e9c8121c16283eea72a363cbd673c5e3edcd2acc",
    "source/sample_submission.csv": "3b0e87ae166b4aea68292fdd1443ee6d436e09a605ce77245587dd4273ab7465",
    "source/baseline_persistence.csv": "0533ef3ad4bdff406a7680c9d4b17033d7d81afddad0cd5579ccbca80ec43110",
    "cache/manifest.json": "33ec27b00a3aaf81a92879d49d1db405b37948c2b50ac5317447f2c3e0225b7e",
    "cache/train_features.parquet": "f974e7951ed9490e68b96154f89afd69ee98e4ed2d27c179fc898779a4aec388",
    "cache/train_anchors.parquet": "07452389a19efd63121f4465a9c08cf7f9ef9e58cf1e3ea1f577e2dca5d8611a",
    "cache/test_features.parquet": "004018935c155b0ab4fea18bdcfa2c99bdef265734c1ddcdd5ea5c2fee68312d",
    "frozen/equal_submission.csv": "77e2bdeed39c898402c8c466a695c88f8e70ce837564d17e4dada3f86f708ab6",
    "frozen/equal_manifest.json": "f260fa35b677ff65aa35ed34f386c9cef3cf53590727673e83245282afa6baa3",
    "frozen/router_submission.csv": "6a6012c711d3b3022102caf33829dfc8c73893bc2e97218e48bbd27bcdeaf289",
    "frozen/router_manifest.json": "539f2acd1afde0f89fa6dd130b901f17f4b2d72031f7c29f0d9d0c68586eff86",
    "frozen/current_submission.csv": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
    "frozen/current_manifest.json": "886637d3f7c92437a6ac40f0aee496d67b44f5913acf75a1502abb566e3c62fd",
    "current/ready_submission.csv": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7"
  },
  "output": {
    "artifact_dir": "artifacts/p3_corrected_repeated_forward_catboost_v2",
    "candidate_relative_path": "candidate/submission.csv",
    "uploaded": false
  },
  "prohibitions": {
    "hyperparameter_search": true,
    "absolute_test_timestamp_recovery": true,
    "test_target_or_hidden_label_access": true,
    "current_or_frozen_submission_mutation": true,
    "submission_upload": true,
    "config_copy_or_arbitrary_output": true,
    "rerun_after_attempt_lock": true,
    "candidate_path_traversal_or_overwrite": true
  }
}
"""
EXPECTED_CONFIG = json.loads(EXPECTED_CONFIG_JSON)


def _now() -> str:
    return base._now()


def _canonical_paths(root: Path) -> tuple[Path, Path, Path]:
    workspace = root.resolve(strict=True)
    return (
        workspace / CANONICAL_CONFIG_RELATIVE,
        workspace / CANONICAL_CACHE_RELATIVE,
        workspace / CANONICAL_OUTPUT_RELATIVE,
    )


def authorize_entry(
    *,
    root: Path,
    requested_config: Path,
    requested_cache: Path,
    requested_output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """First-step authorization used by CLI and direct-call tests."""

    config, paths = authorize_canonical_contract(
        root=root,
        requested_config=requested_config,
        requested_cache=requested_cache,
        requested_output=requested_output,
        canonical_config_relative=CANONICAL_CONFIG_RELATIVE,
        canonical_cache_relative=CANONICAL_CACHE_RELATIVE,
        canonical_output_relative=CANONICAL_OUTPUT_RELATIVE,
        expected_config_sha256=EXPECTED_CONFIG_SHA256,
        expected_config=EXPECTED_CONFIG,
    )
    if config["canonical_paths"] != {
        "config": CANONICAL_CONFIG_RELATIVE,
        "cache": CANONICAL_CACHE_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
    }:
        raise PermissionError("canonical path fields differ from compiled identities")
    if config["model"]["fold_seeds"] != [20260816, 20260817, 20260818]:
        raise PermissionError("fold seeds differ from the compiled contract")
    if config["model"]["full_train_seed"] != 20260817:
        raise PermissionError("full-train seed differs from the compiled contract")
    if config["validation"]["gap_hours"] != 78 or config["validation"]["footprint_hours"] != 72:
        raise PermissionError("split constants differ from the compiled contract")
    if config["router"]["hyperparameter_search"] is not False:
        raise PermissionError("router search must remain disabled")
    implementation_paths = {
        "base_runner": root / "scripts/run_p3_corrected_repeated_forward_catboost_v1.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
    }
    for name, path in implementation_paths.items():
        if base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    return config, paths


def _preflight_authorized(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    cache_dir = paths["cache"]
    pinned_paths = base._resolved_input_paths(root=root, data_dir=data_dir, cache_dir=cache_dir)
    snapshot = base._verify_input_hashes(pinned_paths, config["expected_sha256"])
    if get_gpu_device_count() != 1:
        raise RuntimeError("canonical run requires exactly one visible GPU device")
    features = pd.read_parquet(cache_dir / "train_features.parquet")
    anchors = pd.read_parquet(cache_dir / "train_anchors.parquet")
    if len(features) != len(anchors) or len(anchors) != 24_360:
        raise ValueError("training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("training feature/anchor keys differ")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    if len(feature_columns) != config["features"]["expected_feature_count"]:
        raise ValueError("compact feature count differs")
    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=config["validation"]["gap_hours"],
        footprint_hours=config["validation"]["footprint_hours"],
    )
    aggregate = {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "feature_count": len(feature_columns),
        "validation_cases": len(selected),
        "validation_rows": len(selected) * 6,
        "fold_validation_cases": {fold.name: len(fold.validation_ids) for fold in folds},
        "fold_train_anchors": {fold.name: len(fold.train_ids) for fold in folds},
        "station_min_gap_hours": split_audit["station_global_minimum_gap_hours"],
        "repeated_episode_count": split_audit["repeated_station_episode_count"],
        "footprint_overlap_pairs": split_audit["context48_plus_target24_footprint_overlap_pairs"],
        "canonical_output_absent": not paths["output"].exists(),
        "attempt_lock_absent": not (root / CANONICAL_LOCK_RELATIVE).exists(),
    }
    if len(selected) != 181 or not all(
        [
            aggregate["repeated_episode_count"] == 0,
            aggregate["footprint_overlap_pairs"] == 0,
            aggregate["canonical_output_absent"],
        ]
    ):
        raise ValueError("canonical corrected validation aggregate differs")
    return {
        "pinned_paths": pinned_paths,
        "input_snapshot": snapshot,
        "features": features,
        "anchors": anchors,
        "feature_columns": feature_columns,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "aggregate": aggregate,
    }


def _protected_roots(root: Path, data_dir: Path, cache_dir: Path) -> tuple[Path, ...]:
    return (
        data_dir,
        cache_dir,
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _exclusive_candidate_writer(
    *,
    authorized_target: Path,
    stage: Path,
    protected_roots: tuple[Path, ...],
):
    def write(frame: pd.DataFrame, test_index: pd.DataFrame, path: str | Path) -> Path:
        requested = Path(path).resolve(strict=False)
        checked = safe_new_stage_path(
            stage,
            "candidate/submission.csv",
            protected_roots=protected_roots,
        )
        if requested != authorized_target or checked != authorized_target:
            raise PermissionError("candidate writer received a non-canonical target")
        validate_submission(frame, test_index)
        checked.parent.mkdir(parents=True, exist_ok=False)
        frame.to_csv(
            checked,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            mode="x",
        )
        reread = pd.read_csv(checked)
        validate_submission(reread, test_index)
        return checked

    return write


def _run_after_lock(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    canonical: dict[str, Path],
    attempt_receipt: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _preflight_authorized(root=root, data_dir=data_dir, config=config, paths=canonical)
    print(json.dumps(preflight["aggregate"], ensure_ascii=False), flush=True)
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_corrected_rf_v2_", dir=tmp_root))
    protected = _protected_roots(root, data_dir, canonical["cache"])
    authorized_candidate = safe_new_stage_path(
        stage,
        config["output"]["candidate_relative_path"],
        protected_roots=protected,
    )

    features: pd.DataFrame = preflight["features"]
    anchors: pd.DataFrame = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    selected: pd.DataFrame = preflight["selected"]
    feature_columns: list[str] = preflight["feature_columns"]
    fold_oof: list[pd.DataFrame] = []
    training_receipts: list[dict[str, Any]] = []
    for number, fold in enumerate(folds):
        print(
            json.dumps(
                {"phase": "fit_corrected_fold", "fold": fold.name, "number": number + 1},
                ensure_ascii=False,
            ),
            flush=True,
        )
        current, receipt = base._fit_fold_components(
            fold=fold,
            fold_number=number,
            features=features,
            anchors=anchors,
            feature_columns=feature_columns,
            config=config,
            model_dir=stage / "models/folds",
        )
        fold_oof.append(current)
        training_receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)

    component_oof = pd.concat(fold_oof, ignore_index=True)
    oof, evaluation, router_material = base._evaluate_fixed_structure(
        component_oof=component_oof,
        train_features=features,
        anchors=anchors,
        fold_order=tuple(fold.name for fold in folds),
        config=config,
        split_audit=preflight["split_audit"],
        expected_validation_ids=selected["anchor_id"].to_numpy(dtype=np.int64),
    )
    base._atomic_parquet(stage / "oof.parquet", oof)
    base._atomic_parquet(
        stage / "validation_keys.parquet",
        selected[["fold", "anchor_id", "station", "episode_id"]],
    )
    base._atomic_json(stage / "feature_columns.json", {"columns": feature_columns})
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
    candidate_receipt: dict[str, Any] | None = None
    if evaluation["gate"]["passed"]:
        # Recheck containment and non-existence before full fit and again in the O_EXCL writer.
        if (
            safe_new_stage_path(
                stage,
                config["output"]["candidate_relative_path"],
                protected_roots=protected,
            )
            != authorized_candidate
        ):
            raise PermissionError("candidate target identity changed before full fit")
        guarded_writer = _exclusive_candidate_writer(
            authorized_target=authorized_candidate,
            stage=stage,
            protected_roots=protected,
        )
        original_writer = base.write_submission
        base.write_submission = guarded_writer
        try:
            candidate_receipt, access = base._fit_full_and_infer(
                root=root,
                data_dir=data_dir,
                cache_dir=canonical["cache"],
                stage=stage,
                features=features,
                anchors=anchors,
                feature_columns=feature_columns,
                router_material=router_material,
                config=config,
            )
        finally:
            base.write_submission = original_writer
        if not authorized_candidate.is_file():
            raise RuntimeError("gate passed but exclusive candidate was not created")

    input_after = base._verify_input_hashes(preflight["pinned_paths"], config["expected_sha256"])
    if input_after != preflight["input_snapshot"]:
        raise RuntimeError("source/cache/current/frozen changed during the one-shot run")
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": (
            "CORRECTED_RESEARCH_EVIDENCE_GATE_PASS_CANDIDATE_CREATED_NOT_UPLOADED"
            if evaluation["gate"]["passed"]
            else "CORRECTED_RESEARCH_EVIDENCE_GATE_FAIL_NO_TEST_INFERENCE"
        ),
        "interpretation": (
            "Corrected repeated-forward research evidence; not an official hidden score, "
            "not fresh confirmation, and not upload authorization."
        ),
        "official_scoring_note": (
            "T=0.624165 is the organizer's policy/scoring constant, not a hidden model score."
        ),
        "one_shot_attempt": attempt_receipt,
        "split_audit": preflight["split_audit"],
        "training_receipts": training_receipts,
        **evaluation,
        "candidate_validation": candidate_receipt,
        "access_counters": access,
        "invariants": {
            "canonical_config_path_sha_deep_equal": True,
            "canonical_cache_and_output_paths": True,
            "persistent_o_excl_attempt_lock": True,
            "candidate_resolved_stage_containment": True,
            "candidate_prewrite_existing_and_protected_root_refusal": True,
            "candidate_exclusive_create": candidate_receipt is not None,
            "hyperparameter_search_run": False,
            "external_observations_used": 0,
            "test_absolute_timestamp_recovered": False,
            "test_target_or_hidden_labels_used": 0,
            "current_or_frozen_submission_mutated": False,
            "submission_uploaded": False,
            "source_cache_current_frozen_sha_unchanged": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    base._atomic_json(stage / "metrics.json", metrics)
    implementation_paths = {
        "config": canonical["config"],
        "runner_v2": Path(__file__).resolve(),
        "base_runner_v1_helpers": root / "scripts/run_p3_corrected_repeated_forward_catboost_v1.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "tests_v2": root / "tests/test_p3_corrected_repeated_forward_catboost_v2.py",
        "feature_builder": root / "src/p3_wave/features.py",
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "append_only_generation": True,
        "canonical_contract": {
            "config_path": CANONICAL_CONFIG_RELATIVE,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "full_deep_equality": True,
            "cache_path": CANONICAL_CACHE_RELATIVE,
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt_receipt["sha256"],
        },
        "implementation_sha256": {
            name: base.sha256_file(path) for name, path in implementation_paths.items()
        },
        "git": base._git_state(root),
        "input_sha256_before": preflight["input_snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": base._artifact_hashes(stage),
        "gate_passed": evaluation["gate"]["passed"],
        "candidate_created": candidate_receipt is not None,
        "candidate_uploaded": False,
        "access_counters": access,
        "no_raw_values_in_manifest": True,
    }
    base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")
    if canonical["output"].exists():
        raise FileExistsError("canonical output appeared before final atomic move")
    stage.replace(canonical["output"])
    result = {
        "status": metrics["status"],
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": base.sha256_file(canonical["output"] / "metrics.json"),
        "oof_sha256": base.sha256_file(canonical["output"] / "oof.parquet"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": (
            base.sha256_file(canonical["output"] / "candidate/submission.csv")
            if candidate_receipt is not None
            else None
        ),
        "gate": evaluation["gate"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    """Consume the sole canonical v2 attempt, then execute; no override parameters exist."""

    config_path, cache_path, output_path = _canonical_paths(root)
    config, canonical = authorize_entry(
        root=root,
        requested_config=config_path,
        requested_cache=cache_path,
        requested_output=output_path,
    )
    attempt_receipt = acquire_persistent_attempt_lock(
        root / CANONICAL_LOCK_RELATIVE,
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        config=config,
        canonical=canonical,
        attempt_receipt=attempt_receipt,
    )


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    config_path, cache_path, output_path = _canonical_paths(root)
    config, canonical = authorize_entry(
        root=root,
        requested_config=config_path,
        requested_cache=cache_path,
        requested_output=output_path,
    )
    if (root / CANONICAL_LOCK_RELATIVE).exists():
        raise FileExistsError("v2 one-shot attempt was already consumed")
    return _preflight_authorized(root=root, data_dir=data_dir, config=config, paths=canonical)[
        "aggregate"
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve(strict=True)
    if args.mode == "check-only":
        print(json.dumps(check_only(root=root, data_dir=data_dir), ensure_ascii=False, indent=2))
        return 0
    run_experiment(root=root, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

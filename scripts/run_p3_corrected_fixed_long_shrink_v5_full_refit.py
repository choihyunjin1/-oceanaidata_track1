"""Actually refit the complete saved P3 v4 winner and materialize its candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count

try:
    import scripts.run_p3_corrected_fixed_long_shrink_v4 as v4
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    import run_p3_corrected_fixed_long_shrink_v4 as v4
from p3_wave.corrected_fixed_long_shrink import FixedLongLeadShrinkCalibrator
from p3_wave.loss_router import (
    ComponentLossRouter,
    RouterConfig,
    build_case_router_data,
    expand_case_router_rows,
)
from p3_wave.models import threshold_case_weights
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock
from p3_wave.validation import expand_leads

CONFIG_RELATIVE = "configs/experiments/p3_corrected_fixed_long_shrink_v5_full_refit.json"
CONFIG_SHA256 = "c0855fdbed825ccc93b3340a0f6e5a5f19a8fbac58df9c503b6984241b1e7fc8"
OUTPUT_RELATIVE = "artifacts/p3_corrected_fixed_long_shrink_v5_full_refit"
LOCK_RELATIVE = "artifacts/p3_corrected_fixed_long_shrink_v5_full_refit.ATTEMPT_LOCK.json"
CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
V2_RELATIVE = "artifacts/p3_corrected_repeated_forward_catboost_v2"
V4_RELATIVE = "artifacts/p3_corrected_fixed_long_shrink_v4"


def _expected_inputs(root: Path, data_dir: Path) -> dict[str, tuple[Path, str]]:
    return {
        "config": (root / CONFIG_RELATIVE, CONFIG_SHA256),
        "train_features": (
            root / CACHE_RELATIVE / "train_features.parquet",
            "f974e7951ed9490e68b96154f89afd69ee98e4ed2d27c179fc898779a4aec388",
        ),
        "train_anchors": (
            root / CACHE_RELATIVE / "train_anchors.parquet",
            "07452389a19efd63121f4465a9c08cf7f9ef9e58cf1e3ea1f577e2dca5d8611a",
        ),
        "test_features": (
            root / CACHE_RELATIVE / "test_features.parquet",
            "004018935c155b0ab4fea18bdcfa2c99bdef265734c1ddcdd5ea5c2fee68312d",
        ),
        "feature_columns": (
            root / V2_RELATIVE / "feature_columns.json",
            "49f1e03b43cce691f046027a4c59aac1bcb6846970c32ebf94c4b02382d4fa7a",
        ),
        "corrected_oof": (
            root / V2_RELATIVE / "oof.parquet",
            "eb0af75ec29210254da0d13d1bb8164c0d6b427f4ad5853622144a11fe795f7e",
        ),
        "winner_metrics": (
            root / V4_RELATIVE / "metrics.json",
            "07d5b2fa1c593e0cecccd568d6238d124f06bb36f6400e3d6eb90f5d8c1db603",
        ),
        "winner_candidate": (
            root / V4_RELATIVE / "candidate/submission.csv",
            "607f7cd4ed2c126d5aa4eb6d8130a651ac465a0c88b4e74c112d585c3421d708",
        ),
        "test_index": (
            data_dir / "test_index.csv",
            "004551346ca5be6e3445d8b8e9c8121c16283eea72a363cbd673c5e3edcd2acc",
        ),
        "sample_submission": (
            data_dir / "sample_submission.csv",
            "3b0e87ae166b4aea68292fdd1443ee6d436e09a605ce77245587dd4273ab7465",
        ),
        "baseline_persistence": (
            data_dir / "baseline_persistence.csv",
            "0533ef3ad4bdff406a7680c9d4b17033d7d81afddad0cd5579ccbca80ec43110",
        ),
        "frozen_current": (
            root / v4.CURRENT_RELATIVE,
            "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
        ),
    }


def _verify_inputs(inputs: dict[str, tuple[Path, str]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (path, expected) in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"required pinned input is missing: {name}")
        digest = v4.sha256_file(path)
        if digest != expected:
            raise ValueError(f"pinned input SHA differs: {name}")
        observed[name] = digest
    return observed


def _multi_target(anchors: pd.DataFrame, anchor_ids: np.ndarray) -> np.ndarray:
    lookup = anchors.set_index("anchor_id")
    current = lookup.loc[anchor_ids, "current_hs"].to_numpy(float)
    return np.column_stack(
        [lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(float) - current for lead in v4.LEADS]
    )


def _fit_router(
    oof: pd.DataFrame, features: pd.DataFrame, anchors: pd.DataFrame
) -> ComponentLossRouter:
    case_x, case_meta, case_components, _ = build_case_router_data(oof, features, anchors)
    lookup = anchors.set_index("anchor_id")
    truth = np.column_stack(
        [lookup.loc[case_meta["anchor_id"], f"target_{lead}"].to_numpy(float) for lead in v4.LEADS]
    )
    row_x, _, _, row_losses = expand_case_router_rows(case_x, case_meta, case_components, truth)
    return ComponentLossRouter(
        RouterConfig(
            name="smooth_medium",
            alpha=10.0,
            temperature_multiplier=2.0,
            strength=0.5,
        )
    ).fit(row_x, row_losses)


def _fit_full_models(
    *, root: Path, stage: Path, config: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    started = time.perf_counter()
    cache = root / CACHE_RELATIVE
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    if len(features) != 24_360 or len(anchors) != 24_360:
        raise ValueError("full-training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("full-training cache keys differ")
    columns_payload = json.loads(
        (root / V2_RELATIVE / "feature_columns.json").read_text(encoding="utf-8")
    )
    feature_columns = columns_payload["columns"]
    if len(feature_columns) != 591:
        raise ValueError("full-training feature count differs")
    train_ids = anchors["anchor_id"].to_numpy(np.int64)
    x_single, y_single, meta_single = expand_leads(features, anchors, train_ids, feature_columns)
    seed = int(config["training"]["seed"])
    single = CatBoostRegressor(
        **config["training"]["single"],
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    single_started = time.perf_counter()
    single.fit(
        v4._cat_frame(x_single),
        y_single,
        sample_weight=threshold_case_weights(meta_single["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single_elapsed = time.perf_counter() - single_started
    lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    x_multi = lookup.loc[train_ids, ["station", *feature_columns]].reset_index(drop=True)
    x_multi["station"] = x_multi["station"].astype(str)
    multi = CatBoostRegressor(
        **config["training"]["multi"],
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    multi_started = time.perf_counter()
    multi.fit(
        x_multi,
        _multi_target(anchors, train_ids),
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
        ),
        cat_features=[0],
        verbose=False,
    )
    multi_elapsed = time.perf_counter() - multi_started
    router_started = time.perf_counter()
    oof = pd.read_parquet(root / V2_RELATIVE / "oof.parquet")
    router = _fit_router(oof, features, anchors)
    router_elapsed = time.perf_counter() - router_started
    model_dir = stage / "models/full"
    model_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "single.cbm": model_dir / "single.cbm",
        "multi.cbm": model_dir / "multi.cbm",
        "router.joblib": model_dir / "router.joblib",
        "calibrator.joblib": model_dir / "calibrator.joblib",
        "feature_columns.json": model_dir / "feature_columns.json",
    }
    single.save_model(paths["single.cbm"])
    multi.save_model(paths["multi.cbm"])
    joblib.dump(router, paths["router.joblib"])
    joblib.dump(FixedLongLeadShrinkCalibrator(), paths["calibrator.joblib"])
    shutil.copy2(root / V2_RELATIVE / "feature_columns.json", paths["feature_columns.json"])
    receipt = {
        "actual_fit_counts": {"single_catboost": 1, "multi_catboost": 1, "router": 1},
        "training_anchor_count": len(train_ids),
        "single_training_rows": len(x_single),
        "feature_count": len(feature_columns),
        "seed": seed,
        "elapsed_seconds": {
            "single": single_elapsed,
            "multi": multi_elapsed,
            "router": router_elapsed,
            "total": time.perf_counter() - started,
        },
        "model_sha256": {name: v4.sha256_file(path) for name, path in paths.items()},
    }
    return paths, receipt


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    config_path = (root / CONFIG_RELATIVE).resolve(strict=True)
    if config_path != root.resolve(strict=True) / CONFIG_RELATIVE:
        raise PermissionError("non-canonical config path")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "p3_corrected_fixed_long_shrink_v5_full_refit":
        raise ValueError("experiment identity differs")
    if config["training"]["calibrator"]["persistence_weight"] != 0.25:
        raise ValueError("winner coefficient differs")
    if not all(config["prohibitions"].values()):
        raise ValueError("all prohibitions must remain enabled")
    snapshot = _verify_inputs(_expected_inputs(root, data_dir))
    if get_gpu_device_count() != 1:
        raise RuntimeError("canonical full refit requires exactly one visible GPU")
    _, winner = v4.evaluate_identical_oof(root)
    if not winner["gate"]["passed"]:
        raise RuntimeError("sealed winner gate no longer passes")
    return {
        "status": "CHECK_ONLY_PASS",
        "config_sha256": snapshot["config"],
        "pinned_input_count": len(snapshot),
        "winner_rmse_m": winner["candidate"]["rmse_m"],
        "output_absent": not (root / OUTPUT_RELATIVE).exists(),
        "attempt_lock_absent": not (root / LOCK_RELATIVE).exists(),
        "gpu_device_count": get_gpu_device_count(),
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    preflight = check_only(root=root, data_dir=data_dir)
    if not preflight["output_absent"] or not preflight["attempt_lock_absent"]:
        raise FileExistsError("append-only output or attempt lock already exists")
    inputs = _expected_inputs(root, data_dir)
    before = _verify_inputs(inputs)
    lock_receipt = acquire_persistent_attempt_lock(
        root / LOCK_RELATIVE,
        experiment_id="p3_corrected_fixed_long_shrink_v5_full_refit",
        config_sha256=CONFIG_SHA256,
        created_at=v4._now(),
    )
    config = json.loads((root / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    started = time.perf_counter()
    temp_root = root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_corrected_shrink_v5_refit_", dir=temp_root))
    model_paths, training_receipt = _fit_full_models(root=root, stage=stage, config=config)
    candidate, test_index, summary = v4._infer_candidate(
        data_dir=data_dir, cache_dir=root / CACHE_RELATIVE, model_paths=model_paths
    )
    candidate_path = stage / "candidate/submission.csv"
    v4._write_submission_exclusive(candidate, test_index, candidate_path)
    reproduced, reproduced_index, reproduction_summary = v4._infer_candidate(
        data_dir=data_dir, cache_dir=root / CACHE_RELATIVE, model_paths=model_paths
    )
    reproduced_path = stage / "candidate/reproduced_submission.csv"
    v4._write_submission_exclusive(reproduced, reproduced_index, reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError("fresh full-refit saved-model reproduction is not byte-identical")
    prior_candidate = root / V4_RELATIVE / "candidate/submission.csv"
    full_refit_matches_prior_winner = candidate_path.read_bytes() == prior_candidate.read_bytes()
    _, evaluation = v4.evaluate_identical_oof(root)
    v4._atomic_json(
        stage / "metrics.json",
        {
            "created_at": v4._now(),
            "experiment_id": "p3_corrected_fixed_long_shrink_v5_full_refit",
            "status": "ACTUAL_FULL_REFIT_WINNER_CANDIDATE_CREATED_NOT_UPLOADED",
            "evaluation": evaluation,
            "training_receipt": training_receipt,
            "candidate_validation": {
                "rows": len(candidate),
                "cases": int(candidate["case_id"].nunique()),
                "key_order_exact": True,
                "finite": bool(np.isfinite(candidate["hs_pred"]).all()),
                "range_0_to_30_m": bool(candidate["hs_pred"].between(0.0, 30.0).all()),
                "saved_model_reproduction_byte_identical": True,
                "full_refit_candidate_byte_identical_to_v4_winner": full_refit_matches_prior_winner,
                "prediction_summary": summary,
                "reproduction_summary": reproduction_summary,
                "candidate_sha256": v4.sha256_file(candidate_path),
                "reproduced_candidate_sha256": v4.sha256_file(reproduced_path),
            },
            "access_counters": {
                "test_feature_cache_reads": 2,
                "test_index_reads": 2,
                "test_context_reads": 0,
                "test_target_or_hidden_label_reads": 0,
                "absolute_test_timestamp_recovery_attempts": 0,
                "current_or_frozen_writes": 0,
                "upload_attempts": 0,
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    after = _verify_inputs(inputs)
    if before != after:
        raise RuntimeError("pinned source/cache/winner/current inputs changed during refit")
    implementation = {
        "config": root / CONFIG_RELATIVE,
        "runner": Path(__file__).resolve(),
        "calibrator": root / "src/p3_wave/corrected_fixed_long_shrink.py",
        "v4_inference": root / "scripts/run_p3_corrected_fixed_long_shrink_v4.py",
        "test": root / "tests/test_p3_corrected_fixed_long_shrink_v5.py",
    }
    manifest = {
        "created_at": v4._now(),
        "experiment_id": "p3_corrected_fixed_long_shrink_v5_full_refit",
        "status": "ACTUAL_FULL_REFIT_WINNER_CANDIDATE_CREATED_NOT_UPLOADED",
        "append_only_generation": True,
        "attempt_lock": lock_receipt,
        "git": v4._git_state(root),
        "input_sha256_before": before,
        "input_sha256_after": after,
        "input_unchanged": True,
        "implementation_sha256": {
            name: v4.sha256_file(path) for name, path in implementation.items()
        },
        "output_files": v4._artifact_hashes(stage),
        "actual_fit_counts": training_receipt["actual_fit_counts"],
        "candidate_created": True,
        "candidate_uploaded": False,
        "test_target_or_hidden_labels_used": 0,
        "absolute_test_timestamp_recovered": False,
        "current_or_frozen_mutated": False,
        "no_raw_values_in_manifest": True,
    }
    v4._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = v4.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError("append-only full-refit output appeared before finalize")
    stage.replace(output)
    result = {
        "status": "ACTUAL_FULL_REFIT_WINNER_CANDIDATE_CREATED_NOT_UPLOADED",
        "artifact_dir": OUTPUT_RELATIVE,
        "candidate_sha256": v4.sha256_file(output / "candidate/submission.csv"),
        "reproduced_candidate_sha256": v4.sha256_file(
            output / "candidate/reproduced_submission.csv"
        ),
        "metrics_sha256": v4.sha256_file(output / "metrics.json"),
        "manifest_sha256": v4.sha256_file(output / "manifest.json"),
        "actual_fit_counts": training_receipt["actual_fit_counts"],
        "corrected_oof_rmse_m": evaluation["candidate"]["rmse_m"],
        "delta_m": evaluation["delta_candidate_minus_incumbent_m"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


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

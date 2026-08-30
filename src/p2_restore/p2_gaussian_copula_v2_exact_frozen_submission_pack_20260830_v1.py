"""Exact frozen P2 Gaussian-copula v2 official-query materialization.

This module never reads hidden truth or score.py and never uploads. It refits the
single frozen v2 deployment model and maps it to SHA-pinned official query vectors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.submission import build_submission, validate_submission
from scripts import run_p2_gaussian_copula_conditional_mean_20260830_v1 as engine
from scripts import run_p2_gaussian_copula_conditional_mean_20260830_v2 as repair

EXPERIMENT_ID = "p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1"
KEYS = ["station", "layer", "time"]


class SubmissionPackError(RuntimeError):
    """Raised when a frozen materialization contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SubmissionPackError(f"Expected JSON object: {path}")
    return value


def ensure_external_output_dir(repo_root: Path, output_dir: Path) -> Path:
    root = repo_root.resolve()
    candidate = output_dir.expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return candidate
    raise SubmissionPackError(
        "Submission CSV staging must be outside the repository, reports, and artifacts"
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise SubmissionPackError("experiment_id drifted")
    policy = config["execution_policy"]
    required = {
        "maximum_executions": 1,
        "maximum_copula_fits": 1,
        "maximum_threads": 1,
        "official_observations_test_index_sample_read_authorized": True,
        "official_hidden_truth_read_allowed": False,
        "score_py_read_allowed": False,
        "baseline_read_allowed": False,
        "result_based_retry": False,
        "result_based_tuning": False,
        "upload_authorized_in_this_runner": False,
        "commit": False,
        "push": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise SubmissionPackError(f"execution policy drifted: {key}")
    frozen = config["frozen_recipe"]
    if frozen["deployment_training_outer"] != "2025_nov_dec":
        raise SubmissionPackError("deployment outer drifted")
    if float(frozen["selected_shrinkage"]) != 0.5:
        raise SubmissionPackError("selected shrinkage drifted")
    if int(frozen["inner_search_or_hpo"]) != 0 or frozen["result_based_change"]:
        raise SubmissionPackError("search or result-based change enabled")


def verify_lineage(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    lineage = config["lineage"]
    for role in (
        "base_config",
        "completion_overlay",
        "historical_result",
        "prediction_commitment",
        "independent_qa",
    ):
        record = lineage[role]
        path = repo_root / record["path"]
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise SubmissionPackError(f"lineage hash mismatch: {role}")
        receipts[role] = {"path": record["path"], "sha256": actual}
    anchor = lineage["training_anchor"]
    anchor_path = repo_root / anchor["path"]
    if anchor_path.stat().st_size != int(anchor["bytes"]):
        raise SubmissionPackError("training anchor byte size drifted")
    anchor_sha = sha256_file(anchor_path)
    if anchor_sha != anchor["sha256"]:
        raise SubmissionPackError("training anchor hash drifted")
    receipts["training_anchor"] = {
        "path": anchor["path"],
        "bytes": anchor_path.stat().st_size,
        "sha256": anchor_sha,
    }
    for relative, expected in lineage["source_snapshots"].items():
        actual = sha256_file(repo_root / relative)
        if actual != expected:
            raise SubmissionPackError(f"source snapshot hash mismatch: {relative}")
        receipts[relative] = {"sha256": actual}

    base_config = load_json(repo_root / lineage["base_config"]["path"])
    overlay = load_json(repo_root / lineage["completion_overlay"]["path"])
    result = load_json(repo_root / lineage["historical_result"]["path"])
    commitment = load_json(repo_root / lineage["prediction_commitment"]["path"])
    qa = load_json(repo_root / lineage["independent_qa"]["path"])
    frozen = config["frozen_recipe"]
    selection = commitment["selections"][frozen["deployment_training_outer"]]
    checks = {
        "base_experiment": base_config["experiment_id"]
        == "p2_gaussian_copula_conditional_mean_20260830_v1",
        "overlay_experiment": overlay["experiment_id"]
        == "p2_gaussian_copula_conditional_mean_20260830_v2",
        "overlay_completion_only": overlay["classification"]
        == "COMPLETION_ONLY_PROFILE_SUPPORT_CONTRACT_REPAIR",
        "result_experiment": result["experiment_id"]
        == "p2_gaussian_copula_conditional_mean_20260830_v2",
        "result_comparator": result["comparator"] == frozen["comparator"],
        "result_hash_in_qa": qa["hashes"]["result"]
        == lineage["historical_result"]["sha256"],
        "qa_pass": qa["qa_status"] == "PASS",
        "selected_shrinkage": float(selection["selected_shrinkage"])
        == float(frozen["selected_shrinkage"]),
        "selected_records_hash": selection["records_sha256"]
        == frozen["selected_shrinkage_records_sha256"],
        "training_blocks": commitment["receipts"][frozen["deployment_training_outer"]][
            "training_blocks"
        ]
        == frozen["training_blocks"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SubmissionPackError(f"lineage semantic checks failed: {failed}")
    return {
        "files": receipts,
        "semantic_checks": checks,
        "base_config": base_config,
        "commitment": commitment,
    }


def verify_input_pins(
    config: dict[str, Any],
    p2_dir: Path,
    base_u_path: Path,
    alpha50_path: Path,
    incumbent_path: Path,
) -> dict[str, dict[str, Any]]:
    pins = config["input_pins"]
    files = {
        "observations": (p2_dir / "observations.csv", pins["observations_sha256"]),
        "test_index": (p2_dir / "test_index.csv", pins["test_index_sha256"]),
        "sample_submission": (
            p2_dir / "sample_submission.csv",
            pins["sample_submission_sha256"],
        ),
        "official_base_u": (base_u_path, pins["official_base_u_sha256"]),
        "official_alpha50_reference": (
            alpha50_path,
            pins["official_alpha50_reference_sha256"],
        ),
        "current_official_incumbent": (
            incumbent_path,
            pins["current_official_incumbent_sha256"],
        ),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for role, (path, expected) in files.items():
        if not path.is_file():
            raise SubmissionPackError(f"missing input: {role}")
        actual = sha256_file(path)
        if actual != expected:
            raise SubmissionPackError(f"input hash mismatch: {role}")
        receipts[role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return receipts


def read_keyed_csv(path: Path, *, require_temp: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    expected = KEYS + (["temp"] if require_temp else [])
    if list(frame.columns) != expected:
        raise SubmissionPackError(f"CSV schema mismatch: {path}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise SubmissionPackError(f"CSV keys null or duplicate: {path}")
    if require_temp:
        values = pd.to_numeric(frame["temp"], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise SubmissionPackError(f"CSV temp nonfinite: {path}")
        frame["temp"] = values
    return frame


def validate_query_sources(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    base_u: pd.DataFrame,
    alpha50: pd.DataFrame,
    incumbent: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    required_rows = int(config["submission_contract"]["required_rows"])
    if len(test) != required_rows:
        raise SubmissionPackError("test_index row count drifted")
    for name, frame in {
        "sample": sample,
        "base_u": base_u,
        "alpha50": alpha50,
        "incumbent": incumbent,
    }.items():
        if len(frame) != len(test) or not frame[KEYS].equals(test[KEYS]):
            raise SubmissionPackError(f"official query key/order mismatch: {name}")
    counts = test["layer"].astype(int).value_counts().sort_index().to_dict()
    expected_counts = {
        int(layer): int(rows)
        for layer, rows in config["submission_contract"]["layer_rows"].items()
    }
    if counts != expected_counts:
        raise SubmissionPackError("official layer row counts drifted")
    support = test.groupby("time", sort=False)["layer"].nunique().value_counts().to_dict()
    expected_support_all = {
        3: int(config["submission_contract"]["complete_three_layer_times"]),
        2: int(config["submission_contract"]["partial_two_layer_times"]),
        1: int(config["submission_contract"]["partial_one_layer_times"]),
    }
    expected_support = {
        layer_count: times
        for layer_count, times in expected_support_all.items()
        if times > 0
    }
    if support != expected_support:
        raise SubmissionPackError("official profile support counts drifted")
    return {
        "rows": len(test),
        "layer_rows": {str(key): int(value) for key, value in counts.items()},
        "profile_times_by_target_layer_count": {
            str(key): int(value) for key, value in support.items()
        },
        "keys_nonnull_unique": True,
        "test_sample_base_reference_incumbent_order_match": True,
    }


def fit_exact_frozen_model(
    observations: pd.DataFrame,
    repo_root: Path,
    base_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    frozen = config["frozen_recipe"]
    outer = base_config["folds"][frozen["deployment_training_outer"]]
    if outer["training_blocks"] != frozen["training_blocks"]:
        raise SubmissionPackError("frozen training block order drifted")
    start = engine.base.utc(frozen["train_stop"])
    if start != engine.base.utc(outer["start"]):
        raise SubmissionPackError("frozen train stop drifted")
    stop = engine.base.utc(outer["stop"])
    masked = observations.copy()
    validation_mask = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(engine.TARGET_LAYERS)
    )
    masked.loc[validation_mask, ["temp", "psal"]] = np.nan
    panel, _, _ = engine.build_layer_identity_panel(masked)
    endpoints = engine.public_endpoint_frame(masked)
    anchor_path = repo_root / config["lineage"]["training_anchor"]["path"]
    model_config = engine.reference_config(base_config)
    training_parts: list[pd.DataFrame] = []
    reference_receipts: dict[str, Any] = {}
    for block in frozen["training_blocks"]:
        training = engine.base.add_metadata(
            engine.base.block_anchor(anchor_path, block, include_truth=True), observations
        )
        if not training["time"].lt(start).all():
            raise SubmissionPackError("training label crosses frozen deployment boundary")
        bounds = base_config["block_bounds"][block]
        reference, receipts = engine.base.alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=model_config,
            exclude=(engine.base.utc(bounds[0]), engine.base.utc(bounds[1])),
        )
        training["reference"] = reference
        training["residual"] = (
            training["truth"].to_numpy(np.float64) - reference
        )
        training_parts.append(training)
        reference_receipts[block] = receipts
    training = pd.concat(training_parts, ignore_index=True)
    train_times, train_x, train_y, seasons = engine.profile_design(
        training, base_config, require_response=True
    )
    if train_y is None or len(train_times) != int(frozen["expected_training_profiles"]):
        raise SubmissionPackError("frozen training profile count drifted")
    model = engine.fitted_copula(
        train_x,
        train_y,
        seasons,
        float(frozen["selected_shrinkage"]),
        base_config,
    )
    model_receipt = model.receipt()
    receipt_sha = canonical_sha256(model_receipt)
    if receipt_sha != frozen["expected_refit_model_receipt_sha256"]:
        raise SubmissionPackError(
            "refit model receipt differs from the historical frozen outer model"
        )
    return model, {
        "copula_fits": 1,
        "inner_search_or_hpo": 0,
        "training_profiles": len(train_times),
        "training_blocks": frozen["training_blocks"],
        "selected_shrinkage": frozen["selected_shrinkage"],
        "model_receipt": model_receipt,
        "model_receipt_sha256": receipt_sha,
        "training_reference_receipts_sha256": canonical_sha256(reference_receipts),
        "seasonal_oas_reference_reconstructions": len(reference_receipts),
    }


def materialize_candidate(
    repo_root: Path,
    config: dict[str, Any],
    p2_dir: Path,
    base_u_path: Path,
    alpha50_path: Path,
    incumbent_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_config(config)
    lineage = verify_lineage(repo_root, config)
    input_receipts = verify_input_pins(
        config, p2_dir, base_u_path, alpha50_path, incumbent_path
    )
    test = read_keyed_csv(p2_dir / "test_index.csv", require_temp=False)
    sample = read_keyed_csv(p2_dir / "sample_submission.csv", require_temp=True)
    base_u = read_keyed_csv(base_u_path, require_temp=True)
    alpha50 = read_keyed_csv(alpha50_path, require_temp=True)
    incumbent = read_keyed_csv(incumbent_path, require_temp=True)
    query_receipt = validate_query_sources(
        test, sample, base_u, alpha50, incumbent, config
    )

    observations = engine.base.read_observations(p2_dir / "observations.csv")
    model, fit_receipt = fit_exact_frozen_model(
        observations, repo_root, lineage["base_config"], config
    )

    query_seed = test[["layer", "time"]].copy()
    query_seed["time"] = pd.to_datetime(query_seed["time"], utc=True)
    query_seed["current_blend50"] = base_u["temp"].to_numpy(np.float64)
    query_seed["reference"] = alpha50["temp"].to_numpy(np.float64)
    query = engine.base.add_metadata(query_seed, observations)
    if not query["station"].astype("string").reset_index(drop=True).equals(
        test["station"].reset_index(drop=True)
    ):
        raise SubmissionPackError("observation metadata station differs from test_index")

    query_times, query_x, _, query_seasons = engine.profile_design(
        query, lineage["base_config"], require_response=False
    )
    if len(query_times) != int(
        config["submission_contract"]["complete_three_layer_times"]
    ):
        raise SubmissionPackError("complete official query profile count drifted")
    profile_prediction = model.predict(query_x, query_seasons)
    raw = repair.repaired_row_correction(query, query_times, profile_prediction)
    bounded, cap_receipt = engine.bounded_profile_correction(
        raw,
        np.ones(len(raw), dtype=bool),
        rms_cap=float(config["frozen_recipe"]["correction_rms_cap_c"]),
        p99_cap=float(config["frozen_recipe"]["correction_p99_cap_c"]),
    )
    endpoints = engine.public_endpoint_frame(observations)
    reference = alpha50["temp"].to_numpy(np.float64)
    candidate_values = engine.project_profiles_vectorized(
        query, reference + bounded, endpoints
    ).prediction
    if not np.isfinite(candidate_values).all():
        raise SubmissionPackError("official candidate became nonfinite")

    complete_set = set(query_times)
    partial = ~query["time"].isin(complete_set).to_numpy()
    correction = candidate_values - reference
    partial_max = float(np.max(np.abs(correction[partial]), initial=0.0))
    if partial_max > 1e-12:
        raise SubmissionPackError("incomplete official profile was not exact no-op")

    candidate = build_submission(test, candidate_values)
    validation = validate_submission(candidate, test)
    incumbent_values = incumbent["temp"].to_numpy(np.float64)
    base_values = base_u["temp"].to_numpy(np.float64)

    def delta_receipt(other: np.ndarray) -> dict[str, Any]:
        delta = candidate_values - other
        return {
            "changed_rows": int(np.sum(np.abs(delta) > 1e-12)),
            "rms_c": float(np.sqrt(np.mean(np.square(delta)))),
            "p99_abs_c": float(np.quantile(np.abs(delta), 0.99)),
            "maximum_abs_c": float(np.max(np.abs(delta))),
        }

    return candidate, {
        "lineage": {
            "semantic_checks": lineage["semantic_checks"],
            "file_receipts": lineage["files"],
        },
        "inputs": input_receipts,
        "query_contract": query_receipt,
        "fit": fit_receipt,
        "prediction": {
            "complete_profile_times": len(query_times),
            "incomplete_profile_rows": int(partial.sum()),
            "incomplete_profile_max_abs_correction_c": partial_max,
            "preprojection_cap": cap_receipt,
            "postprojection_correction_rms_c": float(
                np.sqrt(np.mean(np.square(correction)))
            ),
            "postprojection_correction_p99_c": float(
                np.quantile(np.abs(correction), 0.99)
            ),
            "postprojection_correction_max_abs_c": float(
                np.max(np.abs(correction))
            ),
            "difference_vs_alpha50": delta_receipt(reference),
            "difference_vs_current_official_incumbent": delta_receipt(
                incumbent_values
            ),
            "difference_vs_official_base_u": delta_receipt(base_values),
        },
        "validation": validation,
        "official_hidden_truth_rows_read": 0,
        "score_py_reads": 0,
        "baseline_reads": 0,
        "uploads": 0,
    }


def prior_submission_hashes(config: dict[str, Any]) -> set[str]:
    history = config["official_history_and_duplicate_denylist"]
    hashes = set(history["older_submitted_sha256"])
    hashes.update(item["submission_sha256"] for item in history["recent_20260829"])
    return hashes


def duplicate_receipt(candidate_sha256: str, config: dict[str, Any]) -> dict[str, Any]:
    history = config["official_history_and_duplicate_denylist"]
    known = prior_submission_hashes(config)
    duplicate = candidate_sha256 in known
    return {
        "candidate_sha256": candidate_sha256,
        "known_prior_submission_hash_count": len(known),
        "exact_hash_duplicate": duplicate,
        "status": "BLOCK_DUPLICATE_NO_UPLOAD" if duplicate else "PASS_NO_EXACT_HASH_DUPLICATE",
        "recent_official_history": history["recent_20260829"],
        "current_best_public_rmse_c": history["current_best_public_rmse_c"],
        "current_best_points": history["current_best_points"],
        "remaining_p2_submissions_today_at_last_ui_check": history[
            "remaining_p2_submissions_today"
        ],
        "semantic_mapping_limitation": history["semantic_mapping_limitation"],
    }

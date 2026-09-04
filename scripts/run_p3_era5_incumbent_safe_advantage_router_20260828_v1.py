"""Run the one-shot, research-only P3 ERA5 incumbent-safe advantage router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["P3_CATBOOST_THREAD_COUNT"] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.era5_context_transfer import common_feature_columns  # noqa: E402
from p3_wave.era5_safe_advantage_router import (  # noqa: E402
    EXPERIMENT_ID,
    LEADS,
    ROUTER_BASE_FEATURES,
    ROUTER_FEATURES,
    AdvantageRouterError,
    apply_bounded_router,
    attach_truth,
    build_inner_block_plan,
    build_router_rows,
    calibrate_tau,
    evaluate_gate,
    exact_incumbent_fallback,
    file_pin,
    fit_advantage_router,
    prior_fold_support,
    router_support_passes,
    sha256_bytes,
    sha256_file,
    write_json_exclusive,
    write_npy_exclusive,
    write_parquet_exclusive,
)

CONFIG_REL = Path(
    "configs/experiments/p3_era5_incumbent_safe_advantage_router_20260828_v1.json"
)
KEYS = ("fold", "anchor_id", "station", "lead_h")
EXPERT_COLUMNS = (
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "incumbent_prediction",
    "transfer_prediction",
    "local_control_prediction",
    "episode_id",
)
OOF_PREDICTOR_COLUMNS = (
    "prefix_fraction",
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "incumbent_prediction",
)
TRUTH_COLUMNS = ("prefix_fraction", "fold", "anchor_id", "station", "lead_h", "target_hs")
EXPECTED_VERSIONS = {
    "python": "3.12.10",
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "pyarrow": "25.0.1",
    "scikit_learn": "1.9.0",
}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    config: Path
    v2_result: Path
    experts: Path
    anchors: Path
    features: Path
    incumbent_oof: Path
    feature_contract: Path
    era5_feature_implementation: Path
    output: Path
    attempt_lock: Path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdvantageRouterError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise AdvantageRouterError(f"path escapes workspace: {value}") from error
    return candidate


def _load_contract(root: Path = ROOT) -> tuple[dict[str, Any], RunPaths]:
    workspace = root.resolve()
    config_path = (workspace / CONFIG_REL).resolve()
    config = _read_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("hypothesis_count") != 1:
        raise AdvantageRouterError("router experiment identity changed")
    if config.get("research_only") is not True:
        raise AdvantageRouterError("router experiment is not research-only")
    frozen = config["frozen_inputs"]
    access = config["access_and_output"]
    paths = RunPaths(
        root=workspace,
        config=config_path,
        v2_result=_resolve(workspace, frozen["era5_v2_result"]["path"]),
        experts=_resolve(workspace, frozen["sealed_expert_predictions"]["path"]),
        anchors=_resolve(workspace, frozen["train_anchors"]["path"]),
        features=_resolve(workspace, frozen["train_features"]["path"]),
        incumbent_oof=_resolve(workspace, frozen["incumbent_oof"]["path"]),
        feature_contract=_resolve(workspace, frozen["feature_contract"]["path"]),
        era5_feature_implementation=_resolve(
            workspace, frozen["era5_feature_implementation"]["path"]
        ),
        output=_resolve(workspace, access["artifact_dir"]),
        attempt_lock=_resolve(workspace, access["attempt_lock"]),
    )
    return config, paths


def _verify_pin(path: Path, pin: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AdvantageRouterError(f"frozen input is missing: {label}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(pin["bytes"]) or digest != str(pin["sha256"]).lower():
        raise AdvantageRouterError(f"frozen input pin changed: {label}")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": size, "sha256": digest}


def _environment_preflight() -> dict[str, Any]:
    import pyarrow
    import sklearn

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if observed != EXPECTED_VERSIONS:
        raise AdvantageRouterError(
            f"environment version drift: expected={EXPECTED_VERSIONS}, observed={observed}"
        )
    threads = {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "P3_CATBOOST_THREAD_COUNT",
        )
    }
    if any(threads[name] != "2" for name in threads if name != "P3_CATBOOST_THREAD_COUNT"):
        raise AdvantageRouterError("non-CatBoost CPU thread limit changed")
    if threads["P3_CATBOOST_THREAD_COUNT"] != "4":
        raise AdvantageRouterError("CatBoost CPU thread limit changed")
    return {
        "passed": True,
        "sys_executable": sys.executable,
        "versions": observed,
        "threads": threads,
        "catboost_imported_or_fit": False,
    }


def _sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)


def _key_sha(frame: pd.DataFrame) -> str:
    raw = frame.loc[:, KEYS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(raw)


def _read_experts(paths: RunPaths) -> pd.DataFrame:
    schema = tuple(pq.ParquetFile(paths.experts).schema_arrow.names)
    if schema != EXPERT_COLUMNS:
        raise AdvantageRouterError("sealed expert schema changed")
    frame = pd.read_parquet(paths.experts, columns=list(EXPERT_COLUMNS))
    if len(frame) != 1086 or frame.duplicated(list(KEYS)).any():
        raise AdvantageRouterError("sealed expert row contract changed")
    values = frame.loc[:, ["incumbent_prediction", "transfer_prediction"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise AdvantageRouterError("sealed expert predictions are non-finite")
    return _sort_rows(frame)


def _read_incumbent_predictors(paths: RunPaths) -> pd.DataFrame:
    schema = tuple(pq.ParquetFile(paths.incumbent_oof).schema_arrow.names)
    if not set(OOF_PREDICTOR_COLUMNS) <= set(schema) or "target_hs" not in schema:
        raise AdvantageRouterError("frozen incumbent OOF schema changed")
    frame = pd.read_parquet(paths.incumbent_oof, columns=list(OOF_PREDICTOR_COLUMNS))
    frame = frame.loc[frame["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction")
    if len(frame) != 1086 or frame.duplicated(list(KEYS)).any():
        raise AdvantageRouterError("frozen incumbent full-prefix predictors changed")
    return _sort_rows(frame)


def _read_metadata(paths: RunPaths) -> pd.DataFrame:
    frame = pd.read_parquet(
        paths.anchors, columns=["anchor_id", "station", "anchor_time", "current_hs"]
    )
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], utc=True, errors="raise")
    if len(frame) != 24360 or frame["anchor_id"].duplicated().any():
        raise AdvantageRouterError("frozen anchor metadata changed")
    return frame


def _read_router_features(paths: RunPaths) -> pd.DataFrame:
    columns = ["anchor_id", *ROUTER_BASE_FEATURES]
    schema = tuple(pq.ParquetFile(paths.features).schema_arrow.names)
    if not set(columns) <= set(schema):
        raise AdvantageRouterError("frozen feature cache lacks a router column")
    frame = pd.read_parquet(paths.features, columns=columns)
    if len(frame) != 24360 or frame["anchor_id"].duplicated().any():
        raise AdvantageRouterError("frozen router feature rows changed")
    return frame


def _read_truth_for_ids(paths: RunPaths, anchor_ids: Sequence[int]) -> pd.DataFrame:
    ids = tuple(sorted(set(int(value) for value in anchor_ids)))
    if not ids:
        raise AdvantageRouterError("truth release requested no anchor IDs")
    table = ds.dataset(paths.incumbent_oof, format="parquet").to_table(
        columns=list(TRUTH_COLUMNS),
        filter=(ds.field("prefix_fraction") == 1.0) & ds.field("anchor_id").isin(list(ids)),
    )
    frame = table.to_pandas().drop(columns="prefix_fraction")
    if len(frame) != len(ids) * len(LEADS) or frame.duplicated(list(KEYS)).any():
        raise AdvantageRouterError("selective truth release does not cover complete cases")
    return _sort_rows(frame)


def _read_all_truth(paths: RunPaths) -> pd.DataFrame:
    frame = pd.read_parquet(paths.incumbent_oof, columns=list(TRUTH_COLUMNS))
    frame = frame.loc[frame["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction")
    if len(frame) != 1086 or frame.duplicated(list(KEYS)).any():
        raise AdvantageRouterError("full-prefix truth release changed")
    return _sort_rows(frame)


def _verify_feature_contract(config: Mapping[str, Any], paths: RunPaths) -> dict[str, Any]:
    contract = _read_json(paths.feature_contract)
    columns = tuple(str(value) for value in contract.get("columns", ()))
    if columns != common_feature_columns() or len(columns) != 286:
        raise AdvantageRouterError("frozen 286-column feature contract changed")
    digest = hashlib.sha256(
        json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected = str(config["frozen_inputs"]["feature_contract"]["columns_sha256"])
    if digest != expected or not set(ROUTER_BASE_FEATURES) <= set(columns):
        raise AdvantageRouterError("router features are not a strict subset of the frozen 286")
    return {"feature_count": len(columns), "columns_sha256": digest}


def _support_receipt(
    config: Mapping[str, Any], anchors: pd.DataFrame, experts: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, dict[str, tuple[int, ...]]]]:
    outer = config["outer_contract"]
    plan = build_inner_block_plan(
        anchors,
        outer["windows"],
        outer_embargo_hours=int(outer["embargo_hours"]),
        block_days=int(outer["inner_block_days"]),
        block_gap_hours=int(outer["inner_block_gap_hours"]),
    )
    support = prior_fold_support(plan, experts, tuple(outer["fold_order"]))
    receipt: dict[str, Any] = {}
    for fold in outer["fold_order"]:
        receipt[str(fold)] = {
            "blocks": [block.public_dict() for block in plan[str(fold)]],
            "exact_prior_expert_support": {
                name: {
                    "cases": len(ids),
                    "anchor_ids_sha256": hashlib.sha256(
                        np.ascontiguousarray(np.asarray(ids, dtype="<i8")).tobytes()
                    ).hexdigest(),
                }
                for name, ids in support[str(fold)].items()
            },
            "router_support_passed": router_support_passes(support[str(fold)]),
        }
    return receipt, support


def check_only(root: Path = ROOT) -> dict[str, Any]:
    config, paths = _load_contract(root)
    frozen = config["frozen_inputs"]
    hashes = {
        "era5_v2_result": _verify_pin(paths.v2_result, frozen["era5_v2_result"], label="v2 result"),
        "sealed_expert_predictions": _verify_pin(
            paths.experts, frozen["sealed_expert_predictions"], label="sealed experts"
        ),
        "train_anchors": _verify_pin(paths.anchors, frozen["train_anchors"], label="anchors"),
        "train_features": _verify_pin(paths.features, frozen["train_features"], label="features"),
        "incumbent_oof": _verify_pin(
            paths.incumbent_oof, frozen["incumbent_oof"], label="incumbent OOF"
        ),
        "feature_contract": _verify_pin(
            paths.feature_contract, frozen["feature_contract"], label="feature contract"
        ),
        "era5_feature_implementation": _verify_pin(
            paths.era5_feature_implementation,
            frozen["era5_feature_implementation"],
            label="ERA5 feature implementation",
        ),
    }
    v2_result = _read_json(paths.v2_result)
    if v2_result.get("experiment_id") != (
        "p3_era5_context_transfer_dependency_recovery_20260828_v2"
    ):
        raise AdvantageRouterError("v2 result identity changed")
    if v2_result.get("blind_seal", {}).get("sha256") != hashes[
        "sealed_expert_predictions"
    ]["sha256"]:
        raise AdvantageRouterError("v2 result no longer binds the sealed experts")
    if int(v2_result.get("check_only_preflight", {}).get("common_feature_count", 0)) != 286:
        raise AdvantageRouterError("v2 common feature count changed")

    environment = _environment_preflight()
    feature_contract = _verify_feature_contract(config, paths)
    experts = _read_experts(paths)
    incumbent = _read_incumbent_predictors(paths)
    if not experts.loc[:, KEYS].equals(incumbent.loc[:, KEYS]):
        raise AdvantageRouterError("sealed expert and incumbent predictor keys differ")
    if experts["incumbent_prediction"].to_numpy(float).tobytes() != incumbent[
        "incumbent_prediction"
    ].to_numpy(float).tobytes():
        raise AdvantageRouterError("sealed expert incumbent bytes differ from frozen OOF")
    anchors = _read_metadata(paths)
    support_receipt, support = _support_receipt(config, anchors, experts)
    eligible = [fold for fold, blocks in support.items() if router_support_passes(blocks)]
    if eligible != ["2025_h1"]:
        raise AdvantageRouterError(f"predeclared exact-support topology changed: {eligible}")
    feature_schema = tuple(pq.ParquetFile(paths.features).schema_arrow.names)
    if not set(ROUTER_BASE_FEATURES) <= set(feature_schema):
        raise AdvantageRouterError("router base features disappeared")
    return {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.check.v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": "check-only",
        "passed": True,
        "writes": 0,
        "outcome_values_read": 0,
        "model_fits": 0,
        "environment": environment,
        "frozen_hashes": hashes,
        "feature_contract": feature_contract,
        "router_features": list(ROUTER_FEATURES),
        "sealed_expert_rows": int(len(experts)),
        "incumbent_bytes_reproduced": True,
        "support": support_receipt,
        "eligible_outer_folds": eligible,
        "preflight_fallback_outer_folds": [
            fold for fold in config["outer_contract"]["fold_order"] if fold not in eligible
        ],
        "anticipated_catboost_fits": 0,
        "anticipated_ridge_fits": 1,
        "official_paths_accessed": False,
        "research_only": True,
    }


def _create_attempt_lock(paths: RunPaths, config_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha256,
        "execute_exactly_once": True,
        "result_driven_retry_or_retune": False,
        "official_access": False,
        "pid": os.getpid(),
    }
    write_json_exclusive(paths.attempt_lock, payload)
    return payload


def _commit_fold(
    paths: RunPaths,
    fold_index: int,
    fold: str,
    rows: pd.DataFrame,
    candidate: np.ndarray,
    active: np.ndarray,
    *,
    decision: str,
    truth_decodes_before_commitment: int,
) -> dict[str, Any]:
    blind_path = paths.output / "blind" / f"fold_{fold_index:02d}_{fold}.npy"
    prediction_sha = write_npy_exclusive(blind_path, candidate)
    incumbent = rows["incumbent_prediction"].to_numpy(dtype=np.float64)
    inactive = ~np.asarray(active, dtype=bool)
    if candidate[inactive].tobytes() != incumbent[inactive].tobytes():
        raise AdvantageRouterError("fold inactive predictions are not exact incumbent")
    receipt = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.fold_commitment.v1",
        "experiment_id": EXPERIMENT_ID,
        "fold_index": int(fold_index),
        "fold": fold,
        "rows": int(len(rows)),
        "cases": int(rows["anchor_id"].nunique()),
        "key_sha256": _key_sha(rows),
        "blind_prediction": {
            "path": blind_path.relative_to(paths.output).as_posix(),
            "bytes": blind_path.stat().st_size,
            "sha256": prediction_sha,
        },
        "decision": decision,
        "active_rows": int(np.asarray(active, dtype=bool).sum()),
        "inactive_rows_bit_exact_incumbent": True,
        "truth_decodes_before_commitment": int(truth_decodes_before_commitment),
        "current_fold_truth_decodes_before_commitment": 0,
    }
    receipt_path = paths.output / "commitments" / f"fold_{fold_index:02d}_{fold}.json"
    write_json_exclusive(receipt_path, receipt)
    return {**receipt, "commitment": file_pin(receipt_path, root=paths.output)}


def _full_prediction_frame(
    rows: pd.DataFrame, candidate: np.ndarray, active: np.ndarray
) -> pd.DataFrame:
    result = rows.loc[:, EXPERT_COLUMNS].copy()
    result["candidate_prediction"] = np.asarray(candidate, dtype=np.float64)
    result["router_active"] = np.asarray(active, dtype=bool)
    return _sort_rows(result)


def _run_independent_core_qa(paths: RunPaths) -> dict[str, Any]:
    qa_script = paths.root / "scripts" / "qa_p3_era5_incumbent_safe_advantage_router_20260828_v1.py"
    command = [
        sys.executable,
        str(qa_script),
        "--root",
        str(paths.root),
        "--core",
        "--write-receipt",
    ]
    completed = subprocess.run(
        command,
        cwd=paths.root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise AdvantageRouterError(
            "independent core QA failed: " + (completed.stderr or completed.stdout).strip()
        )
    receipt_path = paths.output / "independent_qa.json"
    receipt = _read_json(receipt_path)
    if receipt.get("verdict") != "PASS":
        raise AdvantageRouterError("independent core QA did not pass")
    return receipt


def execute_once(root: Path = ROOT) -> dict[str, Any]:
    config, paths = _load_contract(root)
    if paths.output.exists() or paths.attempt_lock.exists():
        raise FileExistsError("one-shot router attempt was already consumed")
    check = check_only(root)
    lock = _create_attempt_lock(paths, sha256_file(paths.config))
    paths.output.mkdir(parents=True, exist_ok=False)
    (paths.output / "blind").mkdir(exist_ok=False)
    (paths.output / "commitments").mkdir(exist_ok=False)

    experts = _read_experts(paths)
    anchors = _read_metadata(paths)
    features = _read_router_features(paths)
    _support_receipt_public, support = _support_receipt(config, anchors, experts)
    fold_order = tuple(str(value) for value in config["outer_contract"]["fold_order"])

    fold_predictions: list[pd.DataFrame] = []
    fold_commitments: list[dict[str, Any]] = []
    router_model: dict[str, Any] | None = None
    inner_i4_gate: dict[str, Any] | None = None
    ridge_fits = 0
    truth_ids_released_before_complete: set[int] = set()

    for fold_index, fold in enumerate(fold_order):
        outer_rows = _sort_rows(experts.loc[experts["fold"].eq(fold)].copy())
        if not router_support_passes(support[fold]):
            candidate, active = exact_incumbent_fallback(outer_rows)
            decision = "BIT_EXACT_INCUMBENT_FALLBACK_INSUFFICIENT_EXACT_INNER_SUPPORT"
        else:
            if fold != "2025_h1" or fold_index != 2:
                raise AdvantageRouterError("unexpected router-eligible outer fold")
            for previous in fold_order[:fold_index]:
                expected = paths.output / "commitments" / (
                    f"fold_{fold_order.index(previous):02d}_{previous}.json"
                )
                if not expected.is_file():
                    raise AdvantageRouterError("prior outer fold was not committed before truth release")

            fit_ids = (*support[fold]["I1"], *support[fold]["I2"])
            tau_ids = support[fold]["I3"]
            gate_ids = support[fold]["I4"]
            fit_rows = build_router_rows(features, experts, fit_ids)
            tau_rows = build_router_rows(features, experts, tau_ids)
            gate_rows = build_router_rows(features, experts, gate_ids)

            fit_truth = _read_truth_for_ids(paths, fit_ids)
            tau_truth = _read_truth_for_ids(paths, tau_ids)
            truth_ids_released_before_complete.update(int(value) for value in fit_ids)
            truth_ids_released_before_complete.update(int(value) for value in tau_ids)
            fit_rows = attach_truth(fit_rows, fit_truth)
            tau_rows = attach_truth(tau_rows, tau_truth)
            model = fit_advantage_router(fit_rows)
            ridge_fits += 1
            tau = calibrate_tau(model, tau_rows)

            gate_advantage = model.predict(gate_rows.loc[:, ROUTER_FEATURES])
            gate_candidate, gate_active = apply_bounded_router(gate_rows, gate_advantage, tau)
            inner_path = paths.output / "blind" / "inner_i4_2025_h1.npy"
            inner_sha = write_npy_exclusive(inner_path, gate_candidate)
            inner_commitment = {
                "schema_version": "p3_era5_incumbent_safe_advantage_router.inner_i4_commitment.v1",
                "experiment_id": EXPERIMENT_ID,
                "outer_fold": fold,
                "cases": int(gate_rows["anchor_id"].nunique()),
                "rows": int(len(gate_rows)),
                "key_sha256": _key_sha(gate_rows),
                "blind_prediction": {
                    "path": inner_path.relative_to(paths.output).as_posix(),
                    "bytes": inner_path.stat().st_size,
                    "sha256": inner_sha,
                },
                "tau": tau,
                "active_rows": int(gate_active.sum()),
                "truth_decodes_before_commitment": len(truth_ids_released_before_complete)
                * len(LEADS),
                "i4_truth_decodes_before_commitment": 0,
            }
            inner_commitment_path = paths.output / "commitments" / "inner_i4_2025_h1.json"
            write_json_exclusive(inner_commitment_path, inner_commitment)

            gate_truth = _read_truth_for_ids(paths, gate_ids)
            truth_ids_released_before_complete.update(int(value) for value in gate_ids)
            evaluated_gate = attach_truth(gate_rows, gate_truth)
            evaluated_gate["candidate_prediction"] = gate_candidate
            evaluated_gate["router_active"] = gate_active
            inner_i4_gate = evaluate_gate(evaluated_gate, require_fold_consistency=False)

            router_model = {
                "schema_version": "p3_era5_incumbent_safe_advantage_router.model.v1",
                "experiment_id": EXPERIMENT_ID,
                "outer_fold": fold,
                "fit_blocks": ["I1", "I2"],
                "tau_block": "I3",
                "blind_gate_block": "I4",
                "fit_cases": len(set(fit_ids)),
                "tau_cases": len(set(tau_ids)),
                "blind_gate_cases": len(set(gate_ids)),
                "fit_rows": int(len(fit_rows)),
                "tau_rows": int(len(tau_rows)),
                "blind_gate_rows": int(len(gate_rows)),
                "tau": tau,
                "blend_strength": 0.20,
                "model": model.public_dict(),
                "inner_i4_gate": inner_i4_gate,
                "inner_i4_commitment": file_pin(inner_commitment_path, root=paths.output),
            }
            if inner_i4_gate["passed"]:
                current_rows = build_router_rows(
                    features,
                    experts,
                    tuple(outer_rows["anchor_id"].drop_duplicates().astype(int)),
                )
                outer_advantage = model.predict(current_rows.loc[:, ROUTER_FEATURES])
                candidate, active = apply_bounded_router(current_rows, outer_advantage, tau)
                outer_rows = current_rows
                decision = "APPLY_BOUNDED_0_20_ROUTER_AFTER_I4_GATE"
            else:
                candidate, active = exact_incumbent_fallback(outer_rows)
                decision = "BIT_EXACT_INCUMBENT_FALLBACK_I4_GATE_FAILED"

        commitment = _commit_fold(
            paths,
            fold_index,
            fold,
            outer_rows,
            candidate,
            active,
            decision=decision,
            truth_decodes_before_commitment=len(truth_ids_released_before_complete) * len(LEADS),
        )
        fold_commitments.append(commitment)
        fold_predictions.append(_full_prediction_frame(outer_rows, candidate, active))

    combined = _sort_rows(pd.concat(fold_predictions, ignore_index=True))
    if len(combined) != 1086 or combined.duplicated(list(KEYS)).any():
        raise AdvantageRouterError("combined outer prediction surface changed")
    complete = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.predictions_complete.v1",
        "experiment_id": EXPERIMENT_ID,
        "fold_order": list(fold_order),
        "fold_commitments": [item["commitment"] for item in fold_commitments],
        "rows": int(len(combined)),
        "cases": int(combined["anchor_id"].nunique()),
        "ridge_fits": ridge_fits,
        "catboost_fits": 0,
        "truth_ids_released_for_causal_inner_training_before_complete": len(
            truth_ids_released_before_complete
        ),
        "active_rows": int(combined["router_active"].sum()),
        "current_fold_truth_decodes_before_own_commitment": 0,
        "official_access": False,
    }
    complete_path = paths.output / "commitments" / "predictions_complete.json"
    write_json_exclusive(complete_path, complete)
    sealed_path = paths.output / "sealed_outer_predictions.parquet"
    sealed_sha = write_parquet_exclusive(sealed_path, combined)

    if not sealed_path.is_file() or sha256_file(sealed_path) != sealed_sha:
        raise AdvantageRouterError("outer predictions were not durably sealed")
    truth = _read_all_truth(paths)
    evaluated = attach_truth(combined, truth)
    outer_gate = evaluate_gate(evaluated, require_fold_consistency=True)
    if inner_i4_gate is None:
        status = "NO_GO_PREFLIGHT"
    elif not inner_i4_gate["passed"]:
        status = "NO_GO_INNER_I4_GATE"
    elif not outer_gate["passed"]:
        status = "NO_GO_OUTER_GATE"
    else:
        status = "LOCAL_GO_RESEARCH_ONLY_STOPPED_BEFORE_OFFICIAL"

    if router_model is not None:
        write_json_exclusive(paths.output / "router_model.json", router_model)
    metrics = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.metrics.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "fits": {"catboost": 0, "ridge": ridge_fits, "total": ridge_fits},
        "support": check["support"],
        "eligible_outer_folds": check["eligible_outer_folds"],
        "fallback_outer_folds": [
            item["fold"] for item in fold_commitments if item["active_rows"] == 0
        ],
        "fold_commitments": [item["commitment"] for item in fold_commitments],
        "inner_i4_gate": inner_i4_gate,
        "outer_gate": outer_gate,
        "intervention_coverage": float(combined["router_active"].mean()),
        "blind_seal": {
            "path": sealed_path.relative_to(paths.root).as_posix(),
            "rows": int(len(combined)),
            "sha256": sealed_sha,
            "sealed_before_full_outer_truth_attach": True,
        },
        "target_access_audit": {
            "current_fold_target_scalar_decodes_before_own_commitment": 0,
            "causal_prior_case_ids_released_before_predictions_complete": len(
                truth_ids_released_before_complete
            ),
            "full_outer_target_scalar_decodes_after_prediction_seal": 1086,
            "official_or_anonymous_test_value_reads": 0,
        },
    }
    write_json_exclusive(paths.output / "metrics.json", metrics)
    result = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "promotion": "research-only" if outer_gate["passed"] else "no-go",
        "check_only_preflight": check,
        "attempt_lock": {
            **lock,
            "sha256": sha256_file(paths.attempt_lock),
        },
        "metrics": metrics,
        "research_only": True,
        "official_access": False,
        "candidate_or_submission_created": False,
        "upload_count": 0,
    }
    write_json_exclusive(paths.output / "result.json", result)
    readme = (
        "# P3 ERA5 incumbent-safe advantage router\n\n"
        f"- Status: `{status}`\n"
        f"- Ridge fits: `{ridge_fits}`; CatBoost fits: `0`\n"
        f"- Intervention coverage: `{metrics['intervention_coverage']:.6f}`\n"
        f"- Outer delta: `{outer_gate['overall']['delta_m']:+.9f} m`\n"
        "- Official test/sample/submission access: `false`\n\n"
        "The frozen experts, 286-feature contract, and source/local split were not modified. "
        "Every inactive prediction is byte-identical to the incumbent.\n"
    )
    from p3_wave.era5_safe_advantage_router import write_bytes_exclusive

    write_bytes_exclusive(paths.output / "README.md", readme.encode("utf-8"))

    qa = _run_independent_core_qa(paths)

    core_files = {
        path.relative_to(paths.output).as_posix(): file_pin(path, root=paths.output)
        for path in sorted(paths.output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "config": file_pin(paths.config, root=paths.root),
        "attempt_lock": file_pin(paths.attempt_lock, root=paths.root),
        "frozen_inputs": check["frozen_hashes"],
        "core_files": core_files,
        "independent_core_qa": file_pin(
            paths.output / "independent_qa.json", root=paths.output
        ),
        "official_access": False,
        "candidate_or_submission_created": False,
        "upload_count": 0,
    }
    write_json_exclusive(paths.output / "manifest.json", manifest)
    seal = {
        "schema_version": "p3_era5_incumbent_safe_advantage_router.seal.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "manifest": file_pin(paths.output / "manifest.json", root=paths.output),
        "blind_predictions": file_pin(sealed_path, root=paths.output),
        "independent_core_qa": file_pin(
            paths.output / "independent_qa.json", root=paths.output
        ),
        "official_access": False,
        "candidate_or_submission_created": False,
        "upload_count": 0,
    }
    write_json_exclusive(paths.output / "seal.json", seal)
    result["independent_qa"] = qa
    result["manifest"] = file_pin(paths.output / "manifest.json", root=paths.output)
    result["seal"] = file_pin(paths.output / "seal.json", root=paths.output)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute_once(args.root) if args.execute else check_only(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

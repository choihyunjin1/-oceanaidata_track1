"""One-shot Q3/Q4 confirmation of the frozen Sobol trial_18 P1 recipe.

This runner never replays Q2 or searches a threshold.  It verifies the sealed
Q2 selection lineage, fits exactly three fixed seeds on each of Q3 and Q4,
seals both prediction artifacts, and only then opens either historical truth
projection.  The historical surfaces are already exposed, so every outcome is
research-only and no submission artifact is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

EXPERIMENT_ID = "p1_mstcn_sobol_trial18_frozen_confirmation_20260830_v3"
BASE_EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"
LEGACY_EXPERIMENT_ID = "p1_mstcn_sobol_hpo_20260829_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


class ContractError(RuntimeError):
    """Raised when a sealed confirmation contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _exclusive_json(path: Path, value: Any) -> None:
    """Create one JSON file without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace_json(path: Path, value: Any) -> None:
    """Atomically replace only nonterminal progress telemetry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def _config(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = _load_json(path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config.get("status") != "PREREGISTERED_ONE_SHOT_LOCAL_HISTORICAL_RESEARCH_ONLY":
        raise ContractError("preregistration status changed")
    recipe = config.get("frozen_recipe", {})
    trial = recipe.get("trial", {})
    if not (
        trial.get("trial_id") == "trial_18"
        and trial.get("trial_index") == 18
        and trial.get("width") == 512
        and trial.get("batch_size") == 64
        and recipe.get("threshold") == 0.8
        and recipe.get("epoch") == 150
        and recipe.get("seeds") == [20260827, 20260839, 20260863]
        and recipe.get("representation") == "raw_three_seed_ensemble_mean"
    ):
        raise ContractError("frozen trial_18 identity changed")
    confirmation = config.get("confirmation_contract", {})
    if not (
        confirmation.get("phases") == ["q3", "q4"]
        and confirmation.get("maximum_lifetime_fit_count") == 6
        and confirmation.get("seal_both_phase_predictions_before_any_phase_truth_metric")
        and confirmation.get("automatic_retry") is False
        and confirmation.get("result_based_tuning") is False
    ):
        raise ContractError("one-shot confirmation contract changed")
    uncertainty = config["evaluation_contract"]["level_2_uncertainty"]
    if not (
        uncertainty.get("replicates") == 10000
        and uncertainty.get("block_days") == 21
        and uncertainty.get("seed") == 20260830
        and config["evaluation_contract"]["level_1_primary"].get("directional_margin") == 0.0
        and config["evaluation_contract"]["level_1_primary"].get("arbitrary_positive_delta_margin")
        is None
    ):
        raise ContractError("metric-aligned evaluation contract changed")
    if not all(value is True for value in config.get("prohibitions", {}).values()):
        raise ContractError("every prohibition must remain active")
    expected_output = {
        "artifact_directory": f"artifacts/{EXPERIMENT_ID}",
        "attempt_lock": f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json",
        "terminal_result": f"artifacts/{EXPERIMENT_ID}/terminal_result.json",
    }
    for name, expected in expected_output.items():
        if config["output_contract"].get(name) != expected:
            raise ContractError(f"output path changed: {name}")
    return config


def _verify_source_pins(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    pins = {"governing_policy": config["governing_policy"], **config["source_pins"]}
    for name, expected in pins.items():
        path = (root / expected["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ContractError(f"pinned source is absent or escapes the repository: {name}")
        identity = {
            "path": expected["path"],
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        if identity != expected:
            raise ContractError(f"pinned source changed: {name}")
        observed[name] = identity
    return observed


def _load_base(*, root: Path = ROOT) -> Any:
    name = f"{EXPERIMENT_ID}_base"
    if name in sys.modules:
        return sys.modules[name]
    path = root / "scripts" / f"run_{BASE_EXPERIMENT_ID}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load the pinned base runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_torch_threads(torch: Any) -> dict[str, int]:
    if torch.get_num_threads() > 2:
        torch.set_num_threads(2)
    try:
        if torch.get_num_interop_threads() > 1:
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() > 2 or torch.get_num_interop_threads() > 1:
        raise ContractError("torch CPU thread cap is not active")
    return {
        "intraop": int(torch.get_num_threads()),
        "interop": int(torch.get_num_interop_threads()),
    }


def _verify_selection_lineage(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    pins = config["source_pins"]
    design = _load_json(root / pins["legacy_sealed_design"]["path"])
    design_receipt = _load_json(root / pins["legacy_sealed_design_receipt"]["path"])
    gate = _load_json(root / pins["legacy_q2_preconfirm_gate"]["path"])
    aggregate = _load_json(root / pins["legacy_aggregate"]["path"])
    terminal = _load_json(root / pins["legacy_terminal"]["path"])
    q2_receipt = _load_json(root / pins["legacy_q2_top2_receipt"]["path"])
    legacy_qa = _load_json(root / pins["legacy_independent_qa"]["path"])
    points = design.get("points", [])
    selected = [row for row in points if row.get("trial_id") == "trial_18"]
    expected_recipe = config["frozen_recipe"]
    expected_trial = expected_recipe["trial"]
    if len(points) != 32 or len(selected) != 1 or selected[0] != expected_trial:
        raise ContractError("sealed design does not contain the exact frozen trial_18")
    if not (
        design_receipt.get("design_sha256") == pins["legacy_sealed_design"]["sha256"]
        and design_receipt.get("point_count") == 32
        and design_receipt.get("sealed_before_first_fit") is True
    ):
        raise ContractError("sealed design receipt changed")
    selected_recipe = gate.get("selected_recipe", {})
    if not (
        selected_recipe.get("trial") == expected_trial
        and selected_recipe.get("threshold") == expected_recipe["threshold"]
        and selected_recipe.get("epoch") == expected_recipe["epoch"]
        and selected_recipe.get("seeds") == expected_recipe["seeds"]
    ):
        raise ContractError("Q2 selected recipe differs from the preregistered recipe")
    legacy_observation = config["selection_lineage"]["legacy_gate_observation"]
    winner_metrics = gate.get("winner_metrics", {})
    checks = gate.get("checks", {})
    if not (
        gate.get("decision") == "STOP_BEFORE_CONFIRMATION"
        and checks.get("all_monthly_delta_f1_strictly_positive") is True
        and checks.get("pooled_delta_f1_gte_0_003") is False
        and checks.get("anchor_positive_removed_rows_eq_0") is True
        and winner_metrics.get("pooled_delta_f1") == legacy_observation["pooled_delta_f1"]
        and gate.get("anchor_positive_removed_rows") == 0
    ):
        raise ContractError("legacy Q2 stop is not the preregistered fixed-margin-only case")
    if not (
        aggregate == terminal
        and aggregate.get("status") == "NO_GO_PRECONFIRM"
        and aggregate.get("q3_q4_training_started") is False
        and aggregate.get("selected_recipe") == selected_recipe
        and aggregate.get("result_based_rerun_authorized") is False
    ):
        raise ContractError("legacy terminal lineage changed")
    q2_score = root / pins["legacy_q2_top2_score"]["path"]
    if not (
        q2_receipt.get("score_sha256") == _sha256(q2_score)
        and q2_receipt.get("score_bytes") == q2_score.stat().st_size
        and q2_receipt.get("config_sha256") == pins["legacy_hpo_config"]["sha256"]
        and q2_receipt.get("design_sha256") == pins["legacy_sealed_design"]["sha256"]
        and q2_receipt.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
        and q2_receipt.get("official_interface_rows_read") == 0
    ):
        raise ContractError("sealed Q2 top-2 receipt changed")
    if legacy_qa.get("decision") != "PASS" or legacy_qa.get("failed_checks") != []:
        raise ContractError("legacy independent QA is not clean")
    present = [
        path
        for value in config["legacy_namespace_absence_contract"]
        if (path := root / value).exists()
    ]
    if present:
        raise ContractError(f"legacy Q3/Q4 confirmation is no longer untouched: {present[0]}")
    return {
        "legacy_experiment_id": LEGACY_EXPERIMENT_ID,
        "design_points": len(points),
        "selected_trial_id": selected_recipe["trial"]["trial_id"],
        "selected_threshold": selected_recipe["threshold"],
        "selected_epoch": selected_recipe["epoch"],
        "selected_seeds": selected_recipe["seeds"],
        "legacy_q2_pooled_delta_f1": winner_metrics["pooled_delta_f1"],
        "legacy_fixed_delta_gate_passed": checks["pooled_delta_f1_gte_0_003"],
        "legacy_all_months_positive": checks["all_monthly_delta_f1_strictly_positive"],
        "legacy_anchor_positive_removed_rows": gate["anchor_positive_removed_rows"],
        "legacy_q3_q4_training_started": aggregate["q3_q4_training_started"],
        "legacy_confirmatory_artifacts_absent": True,
        "legacy_independent_qa": legacy_qa["decision"],
        "q2_search_replayed": False,
        "threshold_search_replayed": False,
    }


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    config = _config(root=root)
    pins = _verify_source_pins(config, root=root)
    lineage = _verify_selection_lineage(config, root=root)
    policy = _load_json(root / config["governing_policy"]["path"])
    if policy.get("status") != "GOVERNING_FUTURE_RESEARCH_POLICY_NO_OFFICIAL_ACTION":
        raise ContractError("governing metric policy status changed")
    base = _load_base(root=root)
    base_config = base._canonical_config(
        root / "configs" / "experiments" / f"{BASE_EXPERIMENT_ID}.json"
    )
    immutable = base.verify_immutable_inputs(base_config, root=root)
    _np, _pd, torch, _model, _data = base._load_scientific()
    threads = _configure_torch_threads(torch)
    artifact_dir = root / "artifacts" / EXPERIMENT_ID
    attempt_lock = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.preflight.v3",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "config_sha256": _sha256(root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"),
        "runner_sha256": _sha256(Path(__file__)),
        "source_pins": pins,
        "selection_lineage": lineage,
        "immutable_inputs": immutable,
        "torch_threads": threads,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifact_namespace_available": not artifact_dir.exists() and not attempt_lock.exists(),
        "maximum_lifetime_fit_count": 6,
        "q2_fits_authorized": 0,
        "q3_q4_historical_truth_surface_exposed": True,
        "claim_scope": "RESEARCH_ONLY",
        "scientific_row_values_read": 0,
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
        "outlier_rows_hard_deleted": 0,
    }


def _trial_config(
    base: Any, source: dict[str, Any], trial: dict[str, Any], seed: int
) -> dict[str, Any]:
    candidate = base._config_for_capacity(source, width=int(trial["width"]), seed=int(seed))
    candidate["architecture"]["dropout"] = float(trial["dropout"])
    candidate["training"]["batch_size"] = int(trial["batch_size"])
    candidate["training"]["learning_rate"] = float(trial["learning_rate"])
    candidate["training"]["weight_decay"] = float(trial["weight_decay"])
    candidate["training"]["loss_weights"]["row_soft_dice"] = float(trial["row_soft_dice_weight"])
    candidate["training"]["loss_weights"]["truncated_temporal_smoothing"] = float(
        trial["temporal_smoothing_weight"]
    )
    candidate["training"]["loss_weights"]["boundary_bce"] = float(trial["boundary_type_weight"])
    candidate["training"]["loss_weights"]["type_bce"] = float(trial["boundary_type_weight"])
    candidate["training"]["stage_weights"] = list(trial["stage_weights"])
    return candidate


def _write_progress(artifact_dir: Path, *, started: float, **values: Any) -> None:
    payload = {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.progress.v3",
        "experiment_id": EXPERIMENT_ID,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        **values,
        "performance_metrics_exposed": False,
        "official_test_sample_submission_hidden_rows_read": 0,
    }
    _replace_json(artifact_dir / "progress.json", payload)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def _fit_one(
    base: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    trial: dict[str, Any],
    seed: int,
    phase: str,
    device: Any,
    artifact_dir: Path,
    run_started: float,
) -> tuple[Any, dict[str, Any]]:
    _np, _pd, torch, _model_api, _data_api = base._load_scientific()
    candidate = _trial_config(base, source_config, trial, seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base._reset_cuda_peak_memory(torch, device)
    model = base._new_model(training.features.shape[1], candidate, device)
    expected = int(
        source_config["architecture"]["exact_parameter_count_by_width_at_input_165"][
            str(trial["width"])
        ]
    )
    if int(model.trainable_parameter_count) != expected:
        raise ContractError("trial parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["training"]["learning_rate"]),
        weight_decay=float(candidate["training"]["weight_decay"]),
    )
    windows = base._selected_windows(training, candidate)
    positive_weight = base._positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = base._schedule_geometry(candidate, window_count=len(windows))
    global_step = 0
    history: list[dict[str, Any]] = []
    history_path = artifact_dir / "histories" / f"{phase}_trial_18_seed_{seed}.json"
    for epoch in range(1, 151):
        epoch_started = time.perf_counter()
        telemetry, global_step, learning_rate = base._train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=candidate,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        history.append(
            base._history_record(
                epoch=epoch,
                telemetry=telemetry,
                global_step=global_step,
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - epoch_started,
            )
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == 150:
            _replace_json(history_path, history)
            _write_progress(
                artifact_dir,
                started=run_started,
                stage="training",
                phase=phase,
                seed=seed,
                epoch=epoch,
                completed_fits=None,
            )
    blind = base.predict_encoded(
        model,
        holdout,
        base._all_windows(holdout, candidate),
        batch_size=int(candidate["training"]["batch_size"]),
        device=device,
    )
    receipt = {
        "phase": phase,
        "trial_id": "trial_18",
        "seed": seed,
        "epochs": 150,
        "optimizer_steps": int(global_step),
        "width": int(trial["width"]),
        "batch_size": int(trial["batch_size"]),
        "parameter_count": int(model.trainable_parameter_count),
        "history_artifact": {
            "path": history_path.relative_to(artifact_dir).as_posix(),
            "bytes": int(history_path.stat().st_size),
            "sha256": _sha256(history_path),
        },
        "checkpoint_persisted": False,
        "nonfinite_count_total": int(sum(row["nonfinite_count"] for row in history)),
        **base._cuda_peak_memory_receipt(torch, device),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return blind, receipt


def _decode_fixed(
    base: Any, bundle: Any, holdout: Any, source_config: dict[str, Any]
) -> tuple[Any, Any]:
    np, _pd, _torch, _model, _data = base._load_scientific()
    score = base._decoder_row_probability(bundle, source_config)
    proposal = base.decode_long_event_segments(
        score,
        bundle.boundary_probability,
        holdout.layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=base._maximum_segment_rows(source_config),
    ).astype(np.int8, copy=False)
    candidate = base.anchor_preserving_union(holdout.surface.anchor, proposal).astype(np.int8)
    return proposal, candidate


def _seal_blind(
    base: Any,
    artifact_dir: Path,
    *,
    phase: str,
    bundle: Any,
    proposal: Any,
    candidate: Any,
    key_sha256: str,
    config_sha256: str,
    recipe_sha256: str,
    fit_receipts: Sequence[dict[str, Any]],
) -> Path:
    score_path = artifact_dir / f"{phase}_confirmatory_blind.npz"
    score_sha256 = base._atomic_npz(
        score_path,
        row_probability=bundle.row_probability,
        boundary_probability=bundle.boundary_probability,
        type_probability=bundle.type_probability,
        proposal=proposal,
        candidate=candidate,
    )
    receipt = {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.blind.v3",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "score_path": score_path.name,
        "score_bytes": int(score_path.stat().st_size),
        "score_sha256": score_sha256,
        "config_sha256": config_sha256,
        "recipe_sha256": recipe_sha256,
        "ordered_holdout_key_sha256": key_sha256,
        "array_inventory": {
            "row_probability": {
                "shape": list(bundle.row_probability.shape),
                "dtype": str(bundle.row_probability.dtype),
            },
            "boundary_probability": {
                "shape": list(bundle.boundary_probability.shape),
                "dtype": str(bundle.boundary_probability.dtype),
            },
            "type_probability": {
                "shape": list(bundle.type_probability.shape),
                "dtype": str(bundle.type_probability.dtype),
            },
            "proposal": {"shape": list(proposal.shape), "dtype": str(proposal.dtype)},
            "candidate": {"shape": list(candidate.shape), "dtype": str(candidate.dtype)},
        },
        "fit_receipts": list(fit_receipts),
        "same_fold_holdout_truth_columns_opened_before_receipt": 0,
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }
    receipt_path = artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
    _exclusive_json(receipt_path, receipt)
    return receipt_path


def _verify_blind_receipt(
    receipt_path: Path, *, config_sha256: str, recipe_sha256: str, expected_key_sha256: str
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    score_path = receipt_path.parent / receipt.get("score_path", "")
    if not (
        receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("config_sha256") == config_sha256
        and receipt.get("recipe_sha256") == recipe_sha256
        and receipt.get("ordered_holdout_key_sha256") == expected_key_sha256
        and receipt.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
        and receipt.get("official_test_sample_submission_hidden_rows_read") == 0
        and score_path.is_file()
        and receipt.get("score_bytes") == score_path.stat().st_size
        and receipt.get("score_sha256") == _sha256(score_path)
    ):
        raise ContractError(f"blind receipt changed: {receipt_path.name}")
    return receipt


def _load_truth(
    base: Any,
    source_config: dict[str, Any],
    holdout: Any,
    receipt_paths: Sequence[Path],
    *,
    fold: str,
    config_sha256: str,
    recipe_sha256: str,
    root: Path,
) -> Any:
    import pyarrow.dataset as dataset

    expected_key = base._ordered_key_sha(holdout.keys)
    for receipt_path in receipt_paths:
        _verify_blind_receipt(
            receipt_path,
            config_sha256=config_sha256,
            recipe_sha256=recipe_sha256,
            expected_key_sha256=expected_key,
        )
    with base._verified_immutable_read(source_config, "frozen_truth_and_folds", root=root) as path:
        truth = (
            dataset.dataset(path, format="parquet")
            .scanner(
                columns=[*base.KEY_COLUMNS, "label", "anomaly_type", "fold"],
                filter=dataset.field("fold") == fold,
                use_threads=True,
            )
            .to_table()
            .to_pandas()
            .reset_index(drop=True)
        )
    truth, _membership = base._validate_registered_holdout_membership(
        truth, source_config, fold=fold
    )
    if not base._keys_equal(holdout.keys, truth):
        raise ContractError("opened truth keys differ from blind holdout keys")
    return truth


def _load_controls(
    base: Any, config: dict[str, Any], surfaces: Any, *, root: Path
) -> dict[str, Any]:
    import numpy as np

    controls: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        score_pin = config["source_pins"][f"incumbent_{phase}_score"]
        receipt_pin = config["source_pins"][f"incumbent_{phase}_receipt"]
        score_path = root / score_pin["path"]
        receipt = _load_json(root / receipt_pin["path"])
        fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
        if not (
            receipt.get("score_sha256") == _sha256(score_path)
            and receipt.get("ordered_holdout_key_sha256") == surfaces.membership_sha256[fold]
        ):
            raise ContractError(f"incumbent {phase} receipt changed")
        with np.load(score_path, allow_pickle=False) as archive:
            index = np.flatnonzero(archive["epochs"] == 150)
            if len(index) != 1:
                raise ContractError(f"incumbent {phase} e150 control is absent")
            controls[phase] = archive["candidate"][int(index[0])].astype(np.int8, copy=True)
    return controls


def classify_evidence_state(
    *, delta_f1: float, ci90_lower: float, ci90_upper: float, level_0_pass: bool
) -> str:
    if not level_0_pass:
        return "QA_BLOCKED"
    if delta_f1 > 0.0:
        if ci90_lower > 0.0:
            return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if delta_f1 < 0.0 and ci90_upper < 0.0:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def _group_metrics(
    base: Any, truth: Any, control: Any, candidate: Any, groups: Any
) -> dict[str, Any]:
    import numpy as np

    values: dict[str, Any] = {}
    group_array = np.asarray(groups).astype(str)
    for group in sorted(set(group_array.tolist())):
        mask = group_array == group
        incumbent = base.binary_metrics(truth[mask], control[mask])
        challenger = base.binary_metrics(truth[mask], candidate[mask])
        values[group] = {
            "rows": int(mask.sum()),
            "incumbent": incumbent,
            "candidate": challenger,
            "delta_f1": float(challenger["f1"] - incumbent["f1"]),
        }
    return values


def _evaluate(
    base: Any,
    truths: dict[str, Any],
    holds: dict[str, Any],
    candidates: dict[str, Any],
    controls: dict[str, Any],
    config: dict[str, Any],
    *,
    fit_receipts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    folds: dict[str, Any] = {}
    truth_parts: list[Any] = []
    candidate_parts: list[Any] = []
    control_parts: list[Any] = []
    anchor_parts: list[Any] = []
    station_parts: list[Any] = []
    layer_parts: list[Any] = []
    month_parts: list[Any] = []
    type_parts: list[Any] = []
    bootstrap_folds: list[tuple[Any, Any, Any, Any]] = []
    for phase in ("q3", "q4"):
        truth = truths[phase]["label"].to_numpy(dtype=np.int8)
        candidate = np.asarray(candidates[phase], dtype=np.int8)
        control = np.asarray(controls[phase], dtype=np.int8)
        anchor = np.asarray(holds[phase].surface.anchor, dtype=np.int8)
        incumbent_metrics = base.binary_metrics(truth, control)
        candidate_metrics = base.binary_metrics(truth, candidate)
        folds[phase] = {
            "rows": int(len(truth)),
            "incumbent": incumbent_metrics,
            "candidate": candidate_metrics,
            "delta_f1": float(candidate_metrics["f1"] - incumbent_metrics["f1"]),
        }
        truth_parts.append(truth)
        candidate_parts.append(candidate)
        control_parts.append(control)
        anchor_parts.append(anchor)
        keys = holds[phase].surface.keys
        station_parts.append(keys["station"].astype(str).to_numpy())
        layer_parts.append(keys["layer"].astype(str).to_numpy())
        month_parts.append(
            pd.to_datetime(keys["time"], utc=True, format="mixed")
            .dt.tz_convert("Asia/Seoul")
            .dt.strftime("%Y-%m")
            .to_numpy()
        )
        type_parts.append(truths[phase]["anomaly_type"].fillna("").astype(str).to_numpy())
        bootstrap_folds.append((keys, truth, control, candidate))
    y = np.concatenate(truth_parts)
    candidate = np.concatenate(candidate_parts)
    control = np.concatenate(control_parts)
    anchor = np.concatenate(anchor_parts)
    stations = np.concatenate(station_parts)
    layers = np.concatenate(layer_parts)
    months = np.concatenate(month_parts)
    anomaly_types = np.concatenate(type_parts)
    incumbent_metrics = base.binary_metrics(y, control)
    candidate_metrics = base.binary_metrics(y, candidate)
    delta_f1 = float(candidate_metrics["f1"] - incumbent_metrics["f1"])
    uncertainty = config["evaluation_contract"]["level_2_uncertainty"]
    bootstrap = base._paired_day_block_bootstrap(
        bootstrap_folds,
        replicates=int(uncertainty["replicates"]),
        block_days=int(uncertainty["block_days"]),
        seed=int(uncertainty["seed"]),
    )
    anchor_removed = int(np.sum((anchor == 1) & (candidate == 0)))
    added = (candidate == 1) & (control == 0)
    removed = (control == 1) & (candidate == 0)
    added_count = int(added.sum())
    removed_count = int(removed.sum())
    added_precision = float(y[added].mean()) if added_count else None
    f1_half = float(incumbent_metrics["f1"] / 2.0)
    algebra_applicable = removed_count == 0 and added_count > 0
    algebra = {
        "role": "hard sanity only when challenger is a pure add-positive union of the incumbent",
        "applicable": algebra_applicable,
        "incumbent_relative_added_rows": added_count,
        "incumbent_relative_removed_rows": removed_count,
        "added_row_precision": added_precision,
        "required_precision_incumbent_f1_over_2": f1_half,
        "precision_condition_positive": (
            bool(added_precision > f1_half)
            if algebra_applicable and added_precision is not None
            else None
        ),
        "condition_matches_primary_direction": (
            bool((added_precision > f1_half) == (delta_f1 > 0.0))
            if algebra_applicable and added_precision is not None
            else None
        ),
    }
    type_recall: dict[str, Any] = {}
    positive_types = sorted(set(anomaly_types[y == 1].tolist()))
    for anomaly_type in positive_types:
        mask = (y == 1) & (anomaly_types == anomaly_type)
        type_recall[anomaly_type or "<blank>"] = {
            "positive_rows": int(mask.sum()),
            "incumbent_recall": float(control[mask].mean()) if mask.any() else None,
            "candidate_recall": float(candidate[mask].mean()) if mask.any() else None,
            "delta_recall": float(candidate[mask].mean() - control[mask].mean())
            if mask.any()
            else None,
        }
    level_0_checks = {
        "fit_count_exactly_6": len(fit_receipts) == 6,
        "all_fits_epoch_150": all(row.get("epochs") == 150 for row in fit_receipts),
        "all_fit_nonfinite_counts_zero": all(
            row.get("nonfinite_count_total") == 0 for row in fit_receipts
        ),
        "anchor_positive_removed_rows_eq_0": anchor_removed == 0,
        "both_blind_seals_preceded_truth": True,
        "fixed_recipe_no_retry_or_mutation": True,
        "official_test_sample_submission_hidden_rows_read_eq_0": True,
        "csv_created_false": True,
        "upload_performed_false": True,
        "outlier_rows_hard_deleted_eq_0": True,
        "label_1_or_anomaly_events_deleted_eq_0": True,
    }
    level_0_pass = all(level_0_checks.values())
    state = classify_evidence_state(
        delta_f1=delta_f1,
        ci90_lower=float(bootstrap["ci90_lower"]),
        ci90_upper=float(bootstrap["ci90_upper"]),
        level_0_pass=level_0_pass,
    )
    readiness = (
        "RESEARCH_ONLY_EXPOSED_SURFACE_EXPLICIT_USER_AUTHORIZATION_REQUIRED_FOR_OFFICIAL_PROBE"
        if state
        in {
            "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY",
            "EXPLORATORY_CHALLENGER_RESEARCH_ONLY",
        }
        else "NOT_READY"
    )
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.metrics.v3",
        "claim_scope": "RESEARCH_ONLY_EXPOSED_HISTORICAL_Q3_Q4",
        "primary": {
            "metric": "pooled_q3_q4_row_level_binary_micro_f1",
            "rows": int(len(y)),
            "incumbent": incumbent_metrics,
            "candidate": candidate_metrics,
            "delta_f1": delta_f1,
            "directional_margin": 0.0,
            "point_estimate_favorable": delta_f1 > 0.0,
        },
        "uncertainty": bootstrap,
        "evidence_state": state,
        "level_0_hard_validity_checks": level_0_checks,
        "level_0_pass": level_0_pass,
        "folds_diagnostic_only": folds,
        "by_station_diagnostic_only": _group_metrics(base, y, control, candidate, stations),
        "by_layer_diagnostic_only": _group_metrics(base, y, control, candidate, layers),
        "by_kst_month_diagnostic_only": _group_metrics(base, y, control, candidate, months),
        "by_anomaly_type_recall_diagnostic_only": type_recall,
        "incumbent_relative_f1_half_algebra": algebra,
        "raw_anchor_positive_removed_rows": anchor_removed,
        "legacy_fixed_delta_or_all_slice_veto_used_for_decision": False,
        "outlier_rows_hard_deleted": 0,
        "label_1_or_anomaly_events_deleted": 0,
        "candidate_submission_readiness": readiness,
        "official_probe_authorized": False,
    }


def _manifest(artifact_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name in {"artifact_manifest.json", "terminal_result.json"}:
            continue
        files.append(
            {
                "path": path.relative_to(artifact_dir).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.manifest.v3",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count_excluding_manifest_and_terminal": len(files),
        "files": files,
        "official_test_sample_submission_hidden_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def execute(*, expected_runner_sha256: str, root: Path = ROOT) -> dict[str, Any]:
    runner_path = Path(__file__)
    runner_sha256 = _sha256(runner_path)
    if expected_runner_sha256.casefold() != runner_sha256:
        raise ContractError("--expected-runner-sha256 must match reviewed runner bytes")
    artifact_dir = root / "artifacts" / EXPERIMENT_ID
    attempt_lock = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    if artifact_dir.exists() or attempt_lock.exists():
        raise FileExistsError("one-shot artifact namespace already exists")
    preflight = check_only(root=root)
    if not preflight["artifact_namespace_available"] or not preflight["cuda_available"]:
        raise ContractError("confirmation requires an unused namespace and CUDA")
    config = _config(root=root)
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config_sha256 = _sha256(config_path)
    recipe_sha256 = hashlib.sha256(_json_bytes(config["frozen_recipe"])).hexdigest()
    lock = {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.attempt.v3",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_sha256,
        "config_sha256": config_sha256,
        "recipe_sha256": recipe_sha256,
        "one_shot": True,
        "maximum_lifetime_fit_count": 6,
        "automatic_retry_authorized": False,
    }
    _exclusive_json(attempt_lock, lock)
    artifact_dir.mkdir(parents=False, exist_ok=False)
    started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    terminal_path = artifact_dir / "terminal_result.json"
    completed_fits = 0
    try:
        _exclusive_json(artifact_dir / "preflight.json", preflight)
        _exclusive_json(
            artifact_dir / "selected_recipe.json",
            {
                **config["frozen_recipe"],
                "experiment_id": EXPERIMENT_ID,
                "recipe_sha256": recipe_sha256,
                "sealed_before_first_fit": True,
                "q2_search_replayed": False,
                "threshold_search_replayed": False,
            },
        )
        base = _load_base(root=root)
        source_config = base._canonical_config(
            root / "configs" / "experiments" / f"{BASE_EXPERIMENT_ID}.json"
        )
        np, _pd, torch, _model, _data = base._load_scientific()
        _configure_torch_threads(torch)
        device = torch.device("cuda")
        surfaces = base.load_blind_surfaces(source_config, root=root)
        controls = _load_controls(base, config, surfaces, root=root)
        candidates: dict[str, Any] = {}
        holds: dict[str, Any] = {}
        receipts: dict[str, Path] = {}
        fit_receipts: list[dict[str, Any]] = []
        for phase in ("q3", "q4"):
            encoder, training, holdout, split = base._prepare_phase_surfaces(
                surfaces, source_config, phase, root=root
            )
            _exclusive_json(artifact_dir / f"{phase}_split.json", split)
            _exclusive_json(artifact_dir / f"{phase}_encoder.json", base._encoder_receipt(encoder))
            row_sum = np.zeros(holdout.surface.rows, dtype=np.float32)
            boundary_sum = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
            type_sum = np.zeros((holdout.surface.rows, len(base.TYPE_NAMES)), dtype=np.float32)
            phase_receipts: list[dict[str, Any]] = []
            for seed in config["frozen_recipe"]["seeds"]:
                blind, fit_receipt = _fit_one(
                    base,
                    training,
                    holdout,
                    source_config=source_config,
                    trial=config["frozen_recipe"]["trial"],
                    seed=int(seed),
                    phase=phase,
                    device=device,
                    artifact_dir=artifact_dir,
                    run_started=run_started,
                )
                row_sum += blind.row_probability
                boundary_sum += blind.boundary_probability
                type_sum += blind.type_probability
                phase_receipts.append(fit_receipt)
                fit_receipts.append(fit_receipt)
                completed_fits += 1
                _write_progress(
                    artifact_dir,
                    started=run_started,
                    stage="fit_complete",
                    phase=phase,
                    seed=seed,
                    completed_fits=completed_fits,
                    maximum_fits=6,
                )
            bundle = base.PredictionBundle(row_sum / 3.0, boundary_sum / 3.0, type_sum / 3.0)
            proposal, candidate = _decode_fixed(base, bundle, holdout, source_config)
            fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
            receipt_path = _seal_blind(
                base,
                artifact_dir,
                phase=phase,
                bundle=bundle,
                proposal=proposal,
                candidate=candidate,
                key_sha256=surfaces.membership_sha256[fold],
                config_sha256=config_sha256,
                recipe_sha256=recipe_sha256,
                fit_receipts=phase_receipts,
            )
            candidates[phase] = candidate
            holds[phase] = holdout
            receipts[phase] = receipt_path
        if completed_fits != 6:
            raise ContractError("confirmation did not complete exactly six fits")
        if _sha256(config_path) != config_sha256 or _sha256(runner_path) != runner_sha256:
            raise ContractError("config or runner changed during the one-shot execution")
        for phase in ("q3", "q4"):
            fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
            _verify_blind_receipt(
                receipts[phase],
                config_sha256=config_sha256,
                recipe_sha256=recipe_sha256,
                expected_key_sha256=surfaces.membership_sha256[fold],
            )
        both_blind_sealed_at = datetime.now(UTC).isoformat()
        truths = {
            phase: _load_truth(
                base,
                source_config,
                holds[phase],
                [receipts["q3"], receipts["q4"]],
                fold={"q3": "2025_q3", "q4": "2025_q4"}[phase],
                config_sha256=config_sha256,
                recipe_sha256=recipe_sha256,
                root=root,
            )
            for phase in ("q3", "q4")
        }
        metrics = _evaluate(
            base,
            truths,
            holds,
            candidates,
            controls,
            config,
            fit_receipts=fit_receipts,
        )
        metrics["both_blind_phase_receipts_sealed_at_utc"] = both_blind_sealed_at
        metrics["truth_metrics_opened_after_both_blind_seals"] = True
        _exclusive_json(artifact_dir / "confirmatory_metrics.json", metrics)
        elapsed_seconds = time.perf_counter() - run_started
        manifest = _manifest(artifact_dir)
        _exclusive_json(artifact_dir / "artifact_manifest.json", manifest)
        manifest_path = artifact_dir / "artifact_manifest.json"
        terminal = {
            "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.terminal.v3",
            "experiment_id": EXPERIMENT_ID,
            "status": metrics["evidence_state"],
            "claim_scope": metrics["claim_scope"],
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "device": torch.cuda.get_device_name(device),
            "config_sha256": config_sha256,
            "runner_sha256": runner_sha256,
            "recipe_sha256": recipe_sha256,
            "attempt_lock_sha256": _sha256(attempt_lock),
            "artifact_manifest": {
                "path": "artifact_manifest.json",
                "bytes": int(manifest_path.stat().st_size),
                "sha256": _sha256(manifest_path),
            },
            "fit_count": completed_fits,
            "optimizer_steps": int(sum(row["optimizer_steps"] for row in fit_receipts)),
            "checkpoint_files_created": 0,
            "q2_fits": 0,
            "q2_search_replayed": False,
            "threshold_search_replayed": False,
            "selected_trial": "trial_18",
            "threshold": 0.8,
            "epoch": 150,
            "seeds": [20260827, 20260839, 20260863],
            "primary": metrics["primary"],
            "uncertainty": metrics["uncertainty"],
            "level_0_pass": metrics["level_0_pass"],
            "legacy_fixed_delta_or_all_slice_veto_used_for_decision": False,
            "candidate_submission_readiness": metrics["candidate_submission_readiness"],
            "official_test_sample_submission_hidden_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
            "outlier_rows_hard_deleted": 0,
            "label_1_or_anomaly_events_deleted": 0,
            "automatic_retry_authorized": False,
        }
        _exclusive_json(terminal_path, terminal)
        return terminal
    except BaseException as error:
        if artifact_dir.exists() and not terminal_path.exists():
            failure = {
                "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.terminal.v3",
                "experiment_id": EXPERIMENT_ID,
                "status": "FAILED_EXECUTION_NO_RETRY_AUTHORIZED",
                "claim_scope": "NO_PERFORMANCE_CLAIM",
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - run_started,
                "completed_fit_count": completed_fits,
                "error_type": type(error).__name__,
                "error": str(error),
                "official_test_sample_submission_hidden_rows_read": 0,
                "csv_created": False,
                "upload_performed": False,
                "automatic_retry_authorized": False,
            }
            _exclusive_json(terminal_path, failure)
        raise


def run_smoke() -> dict[str, Any]:
    config = _config()
    states = {
        "high": classify_evidence_state(
            delta_f1=0.01, ci90_lower=0.001, ci90_upper=0.02, level_0_pass=True
        ),
        "exploratory": classify_evidence_state(
            delta_f1=0.01, ci90_lower=-0.001, ci90_upper=0.02, level_0_pass=True
        ),
        "harm": classify_evidence_state(
            delta_f1=-0.01, ci90_lower=-0.02, ci90_upper=-0.001, level_0_pass=True
        ),
        "blocked": classify_evidence_state(
            delta_f1=0.01, ci90_lower=0.001, ci90_upper=0.02, level_0_pass=False
        ),
    }
    expected = {
        "high": "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY",
        "exploratory": "EXPLORATORY_CHALLENGER_RESEARCH_ONLY",
        "harm": "PRIMARY_HARM_RESEARCH_ONLY",
        "blocked": "QA_BLOCKED",
    }
    if states != expected:
        raise ContractError("synthetic evidence-state smoke failed")
    return {
        "schema_version": "p1.mstcn_sobol_trial18_frozen_confirmation.smoke.v3",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "states": states,
        "fixed_trial": config["frozen_recipe"]["trial"]["trial_id"],
        "fixed_threshold": config["frozen_recipe"]["threshold"],
        "real_input_rows_read": 0,
        "attempt_lock_created": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_only:
        result = check_only()
    elif args.smoke:
        result = run_smoke()
    else:
        if not args.expected_runner_sha256:
            raise ContractError("--execute requires --expected-runner-sha256")
        result = execute(expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

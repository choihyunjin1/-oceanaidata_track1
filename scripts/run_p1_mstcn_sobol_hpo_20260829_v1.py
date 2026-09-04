"""One-shot historical-only Sobol HPO for the P1 MS-TCN++ e150 family.

The runner reuses the byte-pinned incumbent data, model, windowing, decoder,
and training primitives.  It changes only eight preregistered optimization and
regularization axes.  Every Q2 discovery prediction is sealed before Q2 truth
is opened.  If the fixed Q2 gate passes, both Q3 and Q4 predictions are sealed
before either confirmatory metric is computed.  No deployable file is made.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

EXPERIMENT_ID = "p1_mstcn_sobol_hpo_20260829_v1"
SOURCE_EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SOURCE_RUNNER = ROOT / "scripts" / f"run_{SOURCE_EXPERIMENT_ID}.py"


class ContractError(RuntimeError):
    """Raised when the sealed HPO or leakage contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _config(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config.get("status") != "PREREGISTERED_LOCAL_HISTORICAL_ONLY":
        raise ContractError("experiment is not preregistered")
    design = config["sobol_design"]
    if design != {
        **design,
        "implementation": "scipy.stats.qmc.Sobol",
        "dimensions": 8,
        "scramble": True,
        "seed": 20260829,
        "random_base2_m": 5,
        "point_count": 32,
        "seal_before_first_fit": True,
        "result_adaptive_points_allowed": False,
    }:
        raise ContractError("Sobol design identity changed")
    training = config["training_contract"]
    if not (
        training["discovery_seed"] == 20260827
        and training["additional_seeds"] == [20260839, 20260863]
        and training["stop_epoch"] == 150
        and training["source_schedule_horizon_epochs"] == 300
        and training["threshold_grid"] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        and training["top_k_for_three_seed_q2"] == 2
        and training["torch_cpu_threads_max"] == 2
    ):
        raise ContractError("training contract changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ContractError("every prohibition must remain active")
    return config


def _verify_source_pins(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in config["source_pins"].items():
        path = root / expected["path"]
        if not path.is_file():
            raise ContractError(f"pinned source is absent: {name}")
        identity = {"path": expected["path"], "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if identity != expected:
            raise ContractError(f"pinned source changed: {name}")
        observed[name] = identity
    return observed


def _load_base(*, root: Path = ROOT) -> Any:
    name = f"{EXPERIMENT_ID}_base"
    if name in sys.modules:
        return sys.modules[name]
    path = root / "scripts" / f"run_{SOURCE_EXPERIMENT_ID}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load pinned incumbent runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_torch_threads(torch: Any, config: dict[str, Any]) -> dict[str, int]:
    maximum = int(config["training_contract"]["torch_cpu_threads_max"])
    if torch.get_num_threads() > maximum:
        torch.set_num_threads(maximum)
    try:
        if torch.get_num_interop_threads() > 1:
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() > maximum or torch.get_num_interop_threads() > 1:
        raise ContractError("torch CPU thread cap is not active")
    return {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}


def generate_design(config: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import qmc

    registered = config["sobol_design"]
    mapping = registered["mapping"]
    points = qmc.Sobol(
        d=int(registered["dimensions"]),
        scramble=bool(registered["scramble"]),
        seed=int(registered["seed"]),
    ).random_base2(int(registered["random_base2_m"]))
    if points.shape != (32, 8):
        raise ContractError("Sobol point matrix changed")
    templates = mapping["stage_weight_templates"]
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(points):
        u = [float(value) for value in values]
        width = 256 if u[0] < 0.5 else 512
        template_index = min(len(templates) - 1, int(len(templates) * u[7]))
        rows.append(
            {
                "trial_index": index,
                "trial_id": f"trial_{index:02d}",
                "sobol_u": u,
                "width": width,
                "batch_size": int(mapping["batch_size_by_width"][str(width)]),
                "dropout": float(mapping["dropout_linear"][0] + 0.25 * u[1]),
                "learning_rate": float(mapping["learning_rate_log"][0] * (8.0**u[2])),
                "weight_decay": float(mapping["weight_decay_log"][0] * (1000.0**u[3])),
                "row_soft_dice_weight": float(
                    mapping["row_soft_dice_weight_log"][0] * (4.0**u[4])
                ),
                "temporal_smoothing_weight": float(
                    mapping["temporal_smoothing_weight_linear"][0] + 0.25 * u[5]
                ),
                "boundary_type_weight": float(
                    mapping["tied_boundary_type_weight_linear"][0] + 0.30 * u[6]
                ),
                "stage_weights": [float(value) for value in templates[template_index]],
            }
        )
    if sum(row["width"] == 256 for row in rows) != 16:
        raise ContractError("Sobol categorical width balance changed")
    return {
        "schema_version": "p1.mstcn_sobol_hpo.design.v1",
        "experiment_id": EXPERIMENT_ID,
        "generator": {
            "implementation": registered["implementation"],
            "scipy_version": __import__("scipy").__version__,
            "dimensions": 8,
            "scramble": True,
            "seed": 20260829,
            "random_base2_m": 5,
        },
        "points": rows,
        "result_adaptive_points_allowed": False,
    }


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    config = _config(root=root)
    source_pins = _verify_source_pins(config, root=root)
    base = _load_base(root=root)
    base_config = base._canonical_config(root / "configs" / "experiments" / f"{SOURCE_EXPERIMENT_ID}.json")
    immutable = base.verify_immutable_inputs(base_config, root=root)
    _np, _pd, torch, _model, _data = base._load_scientific()
    threads = _configure_torch_threads(torch, config)
    design = generate_design(config)
    return {
        "schema_version": "p1.mstcn_sobol_hpo.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "config_sha256": _sha256(root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"),
        "runner_sha256": _sha256(Path(__file__)),
        "source_pins": source_pins,
        "immutable_inputs": immutable,
        "design_sha256": hashlib.sha256(_json_bytes(design)).hexdigest(),
        "design_points": len(design["points"]),
        "torch_threads": threads,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifact_namespace_available": not ARTIFACT_DIR.exists() and not ATTEMPT_LOCK.exists(),
        "official_interface_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def _trial_config(base: Any, source: dict[str, Any], row: dict[str, Any], seed: int) -> dict[str, Any]:
    candidate = base._config_for_capacity(source, width=int(row["width"]), seed=int(seed))
    candidate["architecture"]["dropout"] = float(row["dropout"])
    candidate["training"]["learning_rate"] = float(row["learning_rate"])
    candidate["training"]["weight_decay"] = float(row["weight_decay"])
    candidate["training"]["loss_weights"]["row_soft_dice"] = float(
        row["row_soft_dice_weight"]
    )
    candidate["training"]["loss_weights"]["truncated_temporal_smoothing"] = float(
        row["temporal_smoothing_weight"]
    )
    candidate["training"]["loss_weights"]["boundary_bce"] = float(
        row["boundary_type_weight"]
    )
    candidate["training"]["loss_weights"]["type_bce"] = float(row["boundary_type_weight"])
    candidate["training"]["stage_weights"] = list(row["stage_weights"])
    return candidate


def _write_progress(base: Any, artifact_dir: Path, **values: Any) -> None:
    payload = {
        "schema_version": "p1.mstcn_sobol_hpo.progress.v1",
        "experiment_id": EXPERIMENT_ID,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        **values,
        "performance_metrics_exposed": False,
    }
    base._atomic_json(artifact_dir / "progress.json", payload, replace=True)
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
    expected = int(source_config["architecture"]["exact_parameter_count_by_width_at_input_165"][str(trial["width"])])
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
    history_path = artifact_dir / "histories" / f"{phase}_{trial['trial_id']}_seed_{seed}.json"
    for epoch in range(1, 151):
        epoch_started = time.perf_counter()
        telemetry, global_step, lr = base._train_epoch(
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
                learning_rate=lr,
                elapsed_seconds=time.perf_counter() - epoch_started,
            )
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == 150:
            base._atomic_json(history_path, history, replace=True)
            _write_progress(
                base,
                artifact_dir,
                stage=phase,
                trial_id=trial["trial_id"],
                seed=seed,
                epoch=epoch,
                elapsed_seconds=time.perf_counter() - run_started,
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
        "trial_id": trial["trial_id"],
        "seed": seed,
        "epochs": 150,
        "optimizer_steps": global_step,
        "width": int(trial["width"]),
        "batch_size": int(trial["batch_size"]),
        "parameter_count": int(model.trainable_parameter_count),
        "history_artifact": {
            "path": history_path.relative_to(artifact_dir).as_posix(),
            "bytes": history_path.stat().st_size,
            "sha256": _sha256(history_path),
        },
        "checkpoint_persisted": False,
        "nonfinite_count_total": int(sum(row["nonfinite_count"] for row in history)),
        **base._cuda_peak_memory_receipt(torch, device),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return blind, receipt


def _decode_grid(base: Any, bundle: Any, holdout: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    np, _pd, _torch, _model, _data = base._load_scientific()
    thresholds = config["training_contract"]["threshold_grid"]
    proposals: list[Any] = []
    candidates: list[Any] = []
    source_config = base._canonical_config()
    score = base._decoder_row_probability(bundle, source_config)
    for threshold in thresholds:
        proposal = base.decode_long_event_segments(
            score,
            bundle.boundary_probability,
            holdout.layout,
            high_threshold=float(threshold),
            snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
            minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
            maximum_rows=base._maximum_segment_rows(source_config),
        )
        proposals.append(proposal.astype(np.int8, copy=False))
        candidates.append(base.anchor_preserving_union(holdout.surface.anchor, proposal).astype(np.int8))
    return np.stack(proposals), np.stack(candidates)


def _seal_npz(
    base: Any,
    artifact_dir: Path,
    *,
    name: str,
    arrays: dict[str, Any],
    config_sha256: str,
    design_sha256: str,
    key_sha256: str,
    fit_receipts: Sequence[dict[str, Any]],
    truth_columns_read: int = 0,
) -> Path:
    score_path = artifact_dir / f"{name}.npz"
    score_sha = base._atomic_npz(score_path, **arrays)
    receipt = {
        "schema_version": "p1.mstcn_sobol_hpo.blind.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "name": name,
        "score_path": score_path.name,
        "score_bytes": score_path.stat().st_size,
        "score_sha256": score_sha,
        "config_sha256": config_sha256,
        "design_sha256": design_sha256,
        "ordered_holdout_key_sha256": key_sha256,
        "array_inventory": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()
        },
        "fit_receipts": list(fit_receipts),
        "same_fold_holdout_truth_columns_opened_before_receipt": truth_columns_read,
        "official_interface_rows_read": 0,
    }
    receipt_path = artifact_dir / f"{name}_receipt.json"
    base._atomic_json(receipt_path, receipt)
    return receipt_path


def _verify_receipt(receipt_path: Path, *, config_sha256: str, design_sha256: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not (
        receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("config_sha256") == config_sha256
        and receipt.get("design_sha256") == design_sha256
        and receipt.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
        and receipt.get("official_interface_rows_read") == 0
    ):
        raise ContractError("blind receipt contract changed")
    score_path = receipt_path.parent / receipt["score_path"]
    identity = {"bytes": score_path.stat().st_size, "sha256": _sha256(score_path)}
    if identity != {"bytes": receipt["score_bytes"], "sha256": receipt["score_sha256"]}:
        raise ContractError("blind score bytes changed")
    return receipt


def _load_arrays(receipt_path: Path, *, config_sha256: str, design_sha256: str) -> dict[str, Any]:
    import numpy as np

    receipt = _verify_receipt(
        receipt_path, config_sha256=config_sha256, design_sha256=design_sha256
    )
    with np.load(receipt_path.parent / receipt["score_path"], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    inventory = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}
    if inventory != receipt["array_inventory"]:
        raise ContractError("blind array inventory changed")
    return arrays


def _load_truth(
    base: Any,
    source_config: dict[str, Any],
    holdout: Any,
    receipts: Sequence[Path],
    *,
    fold: str,
    config_sha256: str,
    design_sha256: str,
    root: Path,
) -> Any:
    import pyarrow.dataset as dataset

    expected_key = base._ordered_key_sha(holdout.keys)
    for path in receipts:
        receipt = _verify_receipt(path, config_sha256=config_sha256, design_sha256=design_sha256)
        if receipt["ordered_holdout_key_sha256"] != expected_key:
            raise ContractError("blind receipt holdout keys changed")
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
    truth, _membership = base._validate_registered_holdout_membership(truth, source_config, fold=fold)
    if not base._keys_equal(holdout.keys, truth):
        raise ContractError("opened truth keys differ from blind holdout")
    return truth


def _control_candidates(base: Any, config: dict[str, Any], surfaces: Any, *, root: Path) -> dict[str, Any]:
    import numpy as np

    q2 = base.load_sealed_q2_grid(
        root / config["source_pins"]["incumbent_q2_receipt"]["path"]
    )
    capacity = np.flatnonzero((q2.widths == 512) & (q2.epochs == 150))
    threshold = np.flatnonzero(np.isclose(q2.thresholds, 0.8, atol=0.0, rtol=0.0))
    if len(capacity) != 1 or len(threshold) != 1:
        raise ContractError("incumbent Q2 control cell changed")
    controls: dict[str, Any] = {
        "q2": q2.candidate[int(capacity[0]), int(threshold[0])].astype(np.int8, copy=True)
    }
    for phase in ("q3", "q4"):
        path = root / config["source_pins"][f"incumbent_{phase}_curve"]["path"]
        receipt_path = root / config["source_pins"][f"incumbent_{phase}_receipt"]["path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("score_sha256") != _sha256(path):
            raise ContractError(f"incumbent {phase} score receipt changed")
        fold = config["confirmation_contract"]["phases"][(0 if phase == "q3" else 1)]
        _ = fold
        with np.load(path, allow_pickle=False) as archive:
            epochs = archive["epochs"]
            index = np.flatnonzero(epochs == 150)
            if len(index) != 1:
                raise ContractError(f"incumbent {phase} e150 is absent")
            controls[phase] = archive["candidate"][int(index[0])].astype(np.int8, copy=True)
        expected_fold = {"q3": "2025_q3", "q4": "2025_q4"}[phase]
        if receipt.get("ordered_holdout_key_sha256") != surfaces.membership_sha256[expected_fold]:
            raise ContractError(f"incumbent {phase} key surface changed")
    return controls


def _score_candidate(base: Any, truth: Any, keys: Any, candidate: Any, control: Any) -> dict[str, Any]:
    import numpy as np

    y = truth["label"].to_numpy(dtype=np.int8)
    pooled_candidate = base.binary_metrics(y, candidate)
    pooled_control = base.binary_metrics(y, control)
    months = keys["time"].astype(str).str.slice(0, 7).to_numpy()
    monthly: dict[str, Any] = {}
    for month in ("2025-04", "2025-05", "2025-06"):
        mask = months == month
        candidate_score = base.binary_metrics(y[mask], candidate[mask])
        control_score = base.binary_metrics(y[mask], control[mask])
        monthly[month] = {
            "rows": int(mask.sum()),
            "control_f1": control_score["f1"],
            "candidate_f1": candidate_score["f1"],
            "delta_f1": float(candidate_score["f1"] - control_score["f1"]),
        }
    return {
        "control": pooled_control,
        "candidate": pooled_candidate,
        "pooled_delta_f1": float(pooled_candidate["f1"] - pooled_control["f1"]),
        "monthly": monthly,
        "minimum_monthly_delta_f1": min(row["delta_f1"] for row in monthly.values()),
    }


def _rank_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            row["metrics"]["minimum_monthly_delta_f1"],
            row["metrics"]["pooled_delta_f1"],
            -int(row["trial"]["width"]),
            float(row["threshold"]),
            -int(row["trial"]["trial_index"]),
        ),
        reverse=True,
    )


def _evaluate_confirmatory(base: Any, truths: dict[str, Any], holds: dict[str, Any], candidates: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    folds: dict[str, Any] = {}
    pooled_truth: list[Any] = []
    pooled_candidate: list[Any] = []
    pooled_control: list[Any] = []
    removed = 0
    for phase in ("q3", "q4"):
        y = truths[phase]["label"].to_numpy(dtype=np.int8)
        candidate_score = base.binary_metrics(y, candidates[phase])
        control_score = base.binary_metrics(y, controls[phase])
        folds[phase] = {
            "control": control_score,
            "candidate": candidate_score,
            "delta_f1": float(candidate_score["f1"] - control_score["f1"]),
        }
        removed += int(np.sum((holds[phase].surface.anchor == 1) & (candidates[phase] == 0)))
        pooled_truth.append(y)
        pooled_candidate.append(candidates[phase])
        pooled_control.append(controls[phase])
    y_all = np.concatenate(pooled_truth)
    candidate_all = np.concatenate(pooled_candidate)
    control_all = np.concatenate(pooled_control)
    candidate_score = base.binary_metrics(y_all, candidate_all)
    control_score = base.binary_metrics(y_all, control_all)
    pooled_delta = float(candidate_score["f1"] - control_score["f1"])
    checks = {
        "q3_delta_f1_strictly_positive": folds["q3"]["delta_f1"] > 0.0,
        "q4_delta_f1_strictly_positive": folds["q4"]["delta_f1"] > 0.0,
        "pooled_delta_f1_strictly_positive": pooled_delta > 0.0,
        "anchor_positive_removed_rows_eq_0": removed == 0,
    }
    return {
        "schema_version": "p1.mstcn_sobol_hpo.confirmatory.v1",
        "folds": folds,
        "pooled": {"control": control_score, "candidate": candidate_score, "delta_f1": pooled_delta},
        "anchor_positive_removed_rows": removed,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "NO_GO",
        "claim_scope": "retrospective historical transport only; not fresh generalization",
    }


def execute(*, expected_runner_sha256: str, root: Path = ROOT) -> dict[str, Any]:
    runner_sha = _sha256(Path(__file__))
    if expected_runner_sha256.casefold() != runner_sha:
        raise ContractError("--expected-runner-sha256 must match reviewed runner bytes")
    if ARTIFACT_DIR.exists() or ATTEMPT_LOCK.exists():
        raise FileExistsError("one-shot artifact namespace already exists")
    preflight = check_only(root=root)
    if not preflight["cuda_available"]:
        raise ContractError("full HPO requires CUDA")
    config = _config(root=root)
    base = _load_base(root=root)
    source_config = base._canonical_config()
    np, _pd, torch, _model, _data = base._load_scientific()
    _configure_torch_threads(torch, config)
    design = generate_design(config)
    design_sha = hashlib.sha256(_json_bytes(design)).hexdigest()
    if design_sha != preflight["design_sha256"]:
        raise ContractError("design changed after preflight")

    lock = {
        "schema_version": "p1.mstcn_sobol_hpo.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_sha,
        "config_sha256": preflight["config_sha256"],
        "design_sha256": design_sha,
        "one_shot": True,
    }
    base._exclusive_json(ATTEMPT_LOCK, lock)
    ARTIFACT_DIR.mkdir(parents=False, exist_ok=False)
    started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    terminal_path = ARTIFACT_DIR / "terminal_result.json"
    try:
        base._atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
        base._atomic_json(ARTIFACT_DIR / "sealed_design.json", design)
        base._atomic_json(
            ARTIFACT_DIR / "sealed_design_receipt.json",
            {
                "schema_version": "p1.mstcn_sobol_hpo.design_receipt.v1",
                "experiment_id": EXPERIMENT_ID,
                "design_path": "sealed_design.json",
                "design_bytes": (ARTIFACT_DIR / "sealed_design.json").stat().st_size,
                "design_sha256": _sha256(ARTIFACT_DIR / "sealed_design.json"),
                "point_count": 32,
                "sealed_before_first_fit": True,
            },
        )
        if _sha256(ARTIFACT_DIR / "sealed_design.json") != design_sha:
            raise ContractError("sealed design bytes differ from preflight")

        device = torch.device("cuda")
        surfaces = base.load_blind_surfaces(source_config, root=root)
        controls = _control_candidates(base, config, surfaces, root=root)
        q2_encoder, q2_train, q2, q2_split = base._prepare_phase_surfaces(
            surfaces, source_config, "q2", root=root
        )
        base._atomic_json(ARTIFACT_DIR / "q2_split.json", q2_split)
        base._atomic_json(ARTIFACT_DIR / "q2_encoder.json", base._encoder_receipt(q2_encoder))
        rows = q2.surface.rows
        thresholds = np.asarray(config["training_contract"]["threshold_grid"], dtype=np.float64)
        row_probability = np.empty((32, rows), dtype=np.float32)
        boundary_probability = np.empty((32, rows, 2), dtype=np.float32)
        type_probability = np.empty((32, rows, len(base.TYPE_NAMES)), dtype=np.float32)
        proposal = np.empty((32, 7, rows), dtype=np.int8)
        candidate = np.empty((32, 7, rows), dtype=np.int8)
        discovery_receipts: list[dict[str, Any]] = []
        for index, trial in enumerate(design["points"]):
            blind, fit_receipt = _fit_one(
                base,
                q2_train,
                q2,
                source_config=source_config,
                trial=trial,
                seed=int(config["training_contract"]["discovery_seed"]),
                phase="q2_discovery",
                device=device,
                artifact_dir=ARTIFACT_DIR,
                run_started=run_started,
            )
            proposals, candidates = _decode_grid(base, blind, q2, config)
            row_probability[index] = blind.row_probability
            boundary_probability[index] = blind.boundary_probability
            type_probability[index] = blind.type_probability
            proposal[index] = proposals
            candidate[index] = candidates
            discovery_receipts.append(fit_receipt)
            _write_progress(
                base,
                ARTIFACT_DIR,
                stage="q2_discovery",
                completed_trials=index + 1,
                total_trials=32,
                elapsed_seconds=time.perf_counter() - run_started,
            )
        discovery_path = _seal_npz(
            base,
            ARTIFACT_DIR,
            name="q2_discovery_blind",
            arrays={
                "thresholds": thresholds,
                "row_probability": row_probability,
                "boundary_probability": boundary_probability,
                "type_probability": type_probability,
                "proposal": proposal,
                "candidate": candidate,
            },
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
            key_sha256=surfaces.membership_sha256["2025_q2"],
            fit_receipts=discovery_receipts,
        )
        discovery_arrays = _load_arrays(
            discovery_path, config_sha256=preflight["config_sha256"], design_sha256=design_sha
        )
        q2_truth = _load_truth(
            base,
            source_config,
            q2.surface,
            [discovery_path],
            fold="2025_q2",
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
            root=root,
        )
        discovery_records: list[dict[str, Any]] = []
        for trial_index, trial in enumerate(design["points"]):
            for threshold_index, threshold in enumerate(thresholds.tolist()):
                metrics = _score_candidate(
                    base,
                    q2_truth,
                    q2.surface.keys,
                    discovery_arrays["candidate"][trial_index, threshold_index],
                    controls["q2"],
                )
                discovery_records.append(
                    {"trial": trial, "threshold": threshold, "threshold_index": threshold_index, "metrics": metrics}
                )
        ranked_discovery = _rank_records(discovery_records)
        top_trials: list[dict[str, Any]] = []
        for record in ranked_discovery:
            if all(record["trial"]["trial_id"] != row["trial_id"] for row in top_trials):
                top_trials.append(record["trial"])
            if len(top_trials) == 2:
                break
        base._atomic_json(
            ARTIFACT_DIR / "q2_discovery_selection.json",
            {
                "schema_version": "p1.mstcn_sobol_hpo.discovery_selection.v1",
                "surface": "Q2 historical only",
                "records": discovery_records,
                "top_trial_ids": [row["trial_id"] for row in top_trials],
            },
        )

        top_row = np.empty((2, rows), dtype=np.float32)
        top_boundary = np.empty((2, rows, 2), dtype=np.float32)
        top_type = np.empty((2, rows, len(base.TYPE_NAMES)), dtype=np.float32)
        top_proposal = np.empty((2, 7, rows), dtype=np.int8)
        top_candidate = np.empty((2, 7, rows), dtype=np.int8)
        top_fit_receipts: list[dict[str, Any]] = []
        for top_index, trial in enumerate(top_trials):
            source_index = int(trial["trial_index"])
            row_sum = discovery_arrays["row_probability"][source_index].copy()
            boundary_sum = discovery_arrays["boundary_probability"][source_index].copy()
            type_sum = discovery_arrays["type_probability"][source_index].copy()
            for seed in config["training_contract"]["additional_seeds"]:
                blind, fit_receipt = _fit_one(
                    base,
                    q2_train,
                    q2,
                    source_config=source_config,
                    trial=trial,
                    seed=int(seed),
                    phase="q2_top2",
                    device=device,
                    artifact_dir=ARTIFACT_DIR,
                    run_started=run_started,
                )
                row_sum += blind.row_probability
                boundary_sum += blind.boundary_probability
                type_sum += blind.type_probability
                top_fit_receipts.append(fit_receipt)
            bundle = base.PredictionBundle(row_sum / 3.0, boundary_sum / 3.0, type_sum / 3.0)
            proposals, candidates = _decode_grid(base, bundle, q2, config)
            top_row[top_index] = bundle.row_probability
            top_boundary[top_index] = bundle.boundary_probability
            top_type[top_index] = bundle.type_probability
            top_proposal[top_index] = proposals
            top_candidate[top_index] = candidates
        top2_path = _seal_npz(
            base,
            ARTIFACT_DIR,
            name="q2_top2_three_seed_blind",
            arrays={
                "trial_indices": np.asarray([row["trial_index"] for row in top_trials], dtype=np.int16),
                "thresholds": thresholds,
                "row_probability": top_row,
                "boundary_probability": top_boundary,
                "type_probability": top_type,
                "proposal": top_proposal,
                "candidate": top_candidate,
            },
            config_sha256=preflight["config_sha256"],
            design_sha256=design_sha,
            key_sha256=surfaces.membership_sha256["2025_q2"],
            fit_receipts=top_fit_receipts,
        )
        top2_arrays = _load_arrays(
            top2_path, config_sha256=preflight["config_sha256"], design_sha256=design_sha
        )
        final_records: list[dict[str, Any]] = []
        for top_index, trial in enumerate(top_trials):
            for threshold_index, threshold in enumerate(thresholds.tolist()):
                final_records.append(
                    {
                        "trial": trial,
                        "threshold": threshold,
                        "threshold_index": threshold_index,
                        "metrics": _score_candidate(
                            base,
                            q2_truth,
                            q2.surface.keys,
                            top2_arrays["candidate"][top_index, threshold_index],
                            controls["q2"],
                        ),
                    }
                )
        winner = _rank_records(final_records)[0]
        winner_trial = winner["trial"]
        winner_metrics = winner["metrics"]
        winner_top_index = next(
            index for index, row in enumerate(top_trials) if row["trial_id"] == winner_trial["trial_id"]
        )
        winner_candidate = top2_arrays["candidate"][winner_top_index, winner["threshold_index"]]
        removed = int(np.sum((q2.surface.anchor == 1) & (winner_candidate == 0)))
        gate_checks = {
            "all_monthly_delta_f1_strictly_positive": all(
                row["delta_f1"] > 0.0 for row in winner_metrics["monthly"].values()
            ),
            "pooled_delta_f1_gte_0_003": winner_metrics["pooled_delta_f1"] >= 0.003,
            "anchor_positive_removed_rows_eq_0": removed == 0,
        }
        selected_recipe = {
            "schema_version": "p1.mstcn_sobol_hpo.selected_recipe.v1",
            "trial": winner_trial,
            "threshold": float(winner["threshold"]),
            "epoch": 150,
            "seeds": [20260827, 20260839, 20260863],
            "selection_surface": "Q2 historical only",
        }
        preconfirm = {
            "schema_version": "p1.mstcn_sobol_hpo.preconfirm.v1",
            "records": final_records,
            "selected_recipe": selected_recipe,
            "winner_metrics": winner_metrics,
            "anchor_positive_removed_rows": removed,
            "checks": gate_checks,
            "decision": "PASS_TO_CONFIRMATION" if all(gate_checks.values()) else "STOP_BEFORE_CONFIRMATION",
        }
        base._atomic_json(ARTIFACT_DIR / "q2_preconfirm_gate.json", preconfirm)
        if not all(gate_checks.values()):
            aggregate = {
                "schema_version": "p1.mstcn_sobol_hpo.aggregate.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_PRECONFIRM",
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - run_started,
                "device": torch.cuda.get_device_name(device),
                "design_sha256": design_sha,
                "discovery_fits": 32,
                "top2_additional_seed_fits": 4,
                "selected_recipe": selected_recipe,
                "preconfirm_gate": preconfirm,
                "q3_q4_training_started": False,
                "official_interface_rows_read": 0,
                "csv_created": False,
                "upload_performed": False,
                "result_based_rerun_authorized": False,
            }
            base._atomic_json(ARTIFACT_DIR / "aggregate.json", aggregate)
            base._atomic_json(terminal_path, aggregate)
            return aggregate

        base._atomic_json(ARTIFACT_DIR / "selected_recipe.json", selected_recipe)
        confirm_receipts: dict[str, Path] = {}
        confirm_holds: dict[str, Any] = {}
        confirm_candidates: dict[str, Any] = {}
        for phase in ("q3", "q4"):
            encoder, training, holdout, split = base._prepare_phase_surfaces(
                surfaces, source_config, phase, root=root
            )
            base._atomic_json(ARTIFACT_DIR / f"{phase}_split.json", split)
            base._atomic_json(ARTIFACT_DIR / f"{phase}_encoder.json", base._encoder_receipt(encoder))
            row_sum = np.zeros(holdout.surface.rows, dtype=np.float32)
            boundary_sum = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
            type_sum = np.zeros((holdout.surface.rows, len(base.TYPE_NAMES)), dtype=np.float32)
            receipts: list[dict[str, Any]] = []
            for seed in selected_recipe["seeds"]:
                blind, fit_receipt = _fit_one(
                    base,
                    training,
                    holdout,
                    source_config=source_config,
                    trial=winner_trial,
                    seed=int(seed),
                    phase=phase,
                    device=device,
                    artifact_dir=ARTIFACT_DIR,
                    run_started=run_started,
                )
                row_sum += blind.row_probability
                boundary_sum += blind.boundary_probability
                type_sum += blind.type_probability
                receipts.append(fit_receipt)
            bundle = base.PredictionBundle(row_sum / 3.0, boundary_sum / 3.0, type_sum / 3.0)
            proposals, candidates = _decode_grid(base, bundle, holdout, config)
            threshold_index = int(np.flatnonzero(np.isclose(thresholds, selected_recipe["threshold"]))[0])
            receipt_path = _seal_npz(
                base,
                ARTIFACT_DIR,
                name=f"{phase}_confirmatory_blind",
                arrays={
                    "row_probability": bundle.row_probability,
                    "boundary_probability": bundle.boundary_probability,
                    "type_probability": bundle.type_probability,
                    "proposal": proposals[threshold_index],
                    "candidate": candidates[threshold_index],
                },
                config_sha256=preflight["config_sha256"],
                design_sha256=design_sha,
                key_sha256=surfaces.membership_sha256[{"q3": "2025_q3", "q4": "2025_q4"}[phase]],
                fit_receipts=receipts,
            )
            confirm_receipts[phase] = receipt_path
            confirm_holds[phase] = holdout
            confirm_candidates[phase] = candidates[threshold_index]
        for path in confirm_receipts.values():
            _verify_receipt(path, config_sha256=preflight["config_sha256"], design_sha256=design_sha)
        truths = {
            phase: _load_truth(
                base,
                source_config,
                confirm_holds[phase].surface,
                [confirm_receipts[phase]],
                fold={"q3": "2025_q3", "q4": "2025_q4"}[phase],
                config_sha256=preflight["config_sha256"],
                design_sha256=design_sha,
                root=root,
            )
            for phase in ("q3", "q4")
        }
        confirmation = _evaluate_confirmatory(
            base, truths, confirm_holds, confirm_candidates, controls
        )
        base._atomic_json(ARTIFACT_DIR / "confirmatory_metrics.json", confirmation)
        aggregate = {
            "schema_version": "p1.mstcn_sobol_hpo.aggregate.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "PASS_RETROSPECTIVE_ONLY" if confirmation["decision"] == "PASS" else "NO_GO_CONFIRMATORY",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - run_started,
            "device": torch.cuda.get_device_name(device),
            "design_sha256": design_sha,
            "discovery_fits": 32,
            "top2_additional_seed_fits": 4,
            "confirmatory_fits": 6,
            "selected_recipe": selected_recipe,
            "preconfirm_gate": preconfirm,
            "confirmation": confirmation,
            "official_interface_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
            "result_based_rerun_authorized": False,
        }
        base._atomic_json(ARTIFACT_DIR / "aggregate.json", aggregate)
        base._atomic_json(terminal_path, aggregate)
        return aggregate
    except BaseException as error:
        if ARTIFACT_DIR.exists() and not terminal_path.exists():
            base._atomic_json(
                terminal_path,
                {
                    "schema_version": "p1.mstcn_sobol_hpo.terminal.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_EXECUTION_NO_RETRY_AUTHORIZED",
                    "started_at_utc": started_at.isoformat(),
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "official_interface_rows_read": 0,
                    "csv_created": False,
                    "upload_performed": False,
                    "automatic_retry_authorized": False,
                },
            )
        raise


def run_smoke(*, root: Path = ROOT) -> dict[str, Any]:
    config = _config(root=root)
    _verify_source_pins(config, root=root)
    base = _load_base(root=root)
    result = base.run_smoke()
    design = generate_design(config)
    if result.get("result") != "PASS" or len(design["points"]) != 32:
        raise ContractError("synthetic smoke failed")
    return {
        "schema_version": "p1.mstcn_sobol_hpo.smoke.v1",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "base_pipeline": result,
        "design_sha256": hashlib.sha256(_json_bytes(design)).hexdigest(),
        "design_points": 32,
        "real_input_reads": 0,
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

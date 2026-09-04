"""One-shot fixed-strength Group-DRO screen for the P1 MS-TCN family."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p1_qc.mstcn_group_dro import (  # noqa: E402
    changed_row_concentration,
    make_group_ids,
    materialize_group_batch,
    robust_bce_from_rows,
)

EXPERIMENT_ID = "p1_mstcn_group_dro_20260829_v2"
CONFIG_PATH = ROOT / "configs/experiments/p1_mstcn_group_dro_20260829_v2.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
SOBOL_PATH = ROOT / "scripts/run_p1_mstcn_sobol_hpo_20260829_v1.py"

SPEC = importlib.util.spec_from_file_location("p1_mstcn_sobol_hpo_frozen", SOBOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load frozen Sobol runner")
SOBOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOBOL
SPEC.loader.exec_module(SOBOL)


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment id changed")
    if config.get("status") != "PREREGISTERED_ROOT_AUTHORIZED_ONE_SHOT":
        raise ContractError("execution status changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ContractError("every prohibition must remain active")
    return config


def _verify_pins(config: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, record in config["source_pins"].items():
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ContractError(f"source pin changed: {name}")
        observed[name] = {"path": record["path"], "sha256": record["sha256"]}
    preconfirm = json.loads(
        (ROOT / config["source_pins"]["sobol_preconfirm"]["path"]).read_text(encoding="utf-8")
    )
    selected = preconfirm["selected_recipe"]["trial"]
    for key, value in config["selected_source_trial"].items():
        if selected.get(key) != value:
            raise ContractError(f"selected Sobol trial changed: {key}")
    return observed


def _synthetic_group_smoke(
    base: Any, source: dict[str, Any], config: dict[str, Any], torch: Any
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = _candidate_config(base, source, config, int(config["training_contract"]["seeds"][0]))
    model = base._new_model(165, candidate, device)
    generator = torch.Generator(device=device).manual_seed(20260829)
    values = torch.randn((2, 64, 165), generator=generator, device=device)
    valid = torch.ones((2, 64), dtype=torch.bool, device=device)
    event = (torch.rand((2, 64), generator=generator, device=device) > 0.9).float()
    boundary = torch.zeros((2, 64, 2), dtype=torch.float32, device=device)
    kinds = torch.zeros((2, 64, 5), dtype=torch.float32, device=device)
    groups = torch.arange(128, device=device).reshape(2, 64) % 4
    output = model(values, valid_mask=valid)
    total, _output, receipt = _group_robust_loss(
        base,
        output,
        event,
        boundary,
        kinds,
        valid,
        groups,
        candidate=candidate,
        positive_weight=5.0,
        group_count=4,
        strength=float(config["group_objective"]["strength"]),
    )
    if not bool(torch.isfinite(total)):
        raise ContractError("synthetic group objective is non-finite")
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise ContractError("synthetic group objective gradients are invalid")
    result = {
        "device": str(device),
        "shape": [2, 64, 165],
        "group_count": 4,
        "loss_finite": True,
        "gradient_finite": True,
        "parameter_count": int(model.trainable_parameter_count),
        "pooled_stage_bce": float(receipt["pooled_stage_bce"].detach().cpu()),
        "worst_group_stage_bce": float(receipt["worst_group_stage_bce"].detach().cpu()),
        "historical_fit_count": 0,
    }
    del model, output, total
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def check_only() -> dict[str, Any]:
    config = _config()
    pins = _verify_pins(config)
    base = SOBOL._load_base(root=ROOT)
    source = base._canonical_config()
    immutable = base.verify_immutable_inputs(source, root=ROOT)
    _np, _pd, torch, _model, _data = base._load_scientific()
    SOBOL._configure_torch_threads(torch, {"training_contract": config["training_contract"]})
    if int(config["training_contract"]["maximum_lifetime_historical_fit_count"]) != 9:
        raise ContractError("fit budget changed")
    smoke = _synthetic_group_smoke(base, source, config, torch)
    return {
        "schema_version": "p1.mstcn_group_dro.preflight.v2",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "helper_sha256": _sha256(ROOT / "src/p1_qc/mstcn_group_dro.py"),
        "source_pins": pins,
        "immutable_inputs": immutable,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "synthetic_group_smoke": smoke,
        "artifact_namespace_available": not ARTIFACT_DIR.exists() and not ATTEMPT_LOCK.exists(),
        "historical_fit_count": 0,
        "official_interface_rows_read": 0,
        "csv_created": False,
        "upload_performed": False,
    }


def _candidate_config(base: Any, source: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, Any]:
    trial = config["selected_source_trial"]
    candidate = SOBOL._trial_config(base, source, trial, seed)
    candidate["training"]["weight_decay"] = float(
        config["training_contract"]["strong_weight_decay"]
    )
    return candidate


def _group_robust_loss(
    base: Any,
    output: Any,
    event: Any,
    boundary: Any,
    kinds: Any,
    valid: Any,
    group_tensor: Any,
    *,
    candidate: dict[str, Any],
    positive_weight: float,
    group_count: int,
    strength: float,
) -> tuple[Any, Any, dict[str, Any]]:
    _np, _pd, torch, model_api, _data = base._load_scientific()
    loss_config = base._loss_config(candidate, positive_weight)
    output_loss = model_api.compute_ms_tcn_asrf_loss(
        output, event, boundary, kinds, valid_mask=valid, config=loss_config
    )
    normalized = [value / sum(loss_config.stage_weights) for value in loss_config.stage_weights]
    correction = output.final_logits.float().sum() * 0.0
    worst_values: list[Any] = []
    pooled_values: list[Any] = []
    pos_weight = output.final_logits.new_tensor(float(positive_weight), dtype=torch.float32)
    for stage_weight, logits in zip(normalized, output.stage_logits, strict=True):
        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), event.float(), pos_weight=pos_weight, reduction="none"
        )
        robust, receipt = robust_bce_from_rows(
            per_row,
            group_tensor,
            valid,
            group_count=group_count,
            strength=strength,
        )
        correction = correction + float(stage_weight) * (
            robust - receipt["pooled_bce"]
        )
        worst_values.append(receipt["worst_group_bce"])
        pooled_values.append(receipt["pooled_bce"])
    total = output_loss.total + loss_config.event_bce_weight * correction
    return total, output_loss, {
        "pooled_stage_bce": sum(float(weight) * value for weight, value in zip(normalized, pooled_values, strict=True)),
        "worst_group_stage_bce": sum(float(weight) * value for weight, value in zip(normalized, worst_values, strict=True)),
    }


def _fit_one(
    base: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    phase: str,
    device: Any,
    run_started: float,
) -> tuple[Any, dict[str, Any]]:
    np, _pd, torch, _model_api, _data_api = base._load_scientific()
    candidate = _candidate_config(base, source_config, config, seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base._reset_cuda_peak_memory(torch, device)
    model = base._new_model(training.features.shape[1], candidate, device)
    expected = int(
        source_config["architecture"]["exact_parameter_count_by_width_at_input_165"][
            str(config["selected_source_trial"]["width"])
        ]
    )
    if int(model.trainable_parameter_count) != expected:
        raise ContractError("parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["training"]["learning_rate"]),
        weight_decay=float(candidate["training"]["weight_decay"]),
    )
    windows = base._selected_windows(training, candidate)
    positive_weight = base._positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = base._schedule_geometry(candidate, window_count=len(windows))
    group_ids, group_receipt = make_group_ids(
        training.surface.keys,
        minimum_rows=int(config["group_objective"]["minimum_rows_before_sparse_merge"]),
    )
    group_count = int(group_receipt["effective_group_count"])
    global_step = 0
    history: list[dict[str, Any]] = []
    history_path = ARTIFACT_DIR / "histories" / f"{phase}_seed_{seed}.json"
    for epoch in range(1, int(config["training_contract"]["epochs"]) + 1):
        epoch_started = time.perf_counter()
        batches = list(
            base._batches(
                windows,
                int(candidate["training"]["batch_size"]),
                seed=int(seed) + epoch,
                shuffle=True,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        sums = {"total_loss": 0.0, "pooled_stage_bce": 0.0, "worst_group_stage_bce": 0.0}
        observed = 0
        grad_norms: list[float] = []
        steps_before = global_step
        accumulation = int(candidate["training"]["gradient_accumulation_steps"])
        _spe, registered_total, warmup_steps = base._schedule_geometry(
            candidate, window_count=len(windows)
        )
        if registered_total != total_steps:
            raise ContractError("LR schedule horizon changed")
        model.train()
        for batch_index, batch_windows in enumerate(batches):
            values, feature_valid = _data_api.materialize_windows(training.features, batch_windows)
            targets = base._materialize_target_batch(training.targets, batch_windows)
            tensor = torch.from_numpy(values).to(device, non_blocking=True)
            event, boundary, kinds, valid = base._move_targets(targets, device)
            feature_valid_tensor = torch.from_numpy(feature_valid).to(
                device, dtype=torch.bool, non_blocking=True
            )
            group_values, group_valid = materialize_group_batch(group_ids, batch_windows)
            if not np.array_equal(feature_valid, group_valid):
                raise ContractError("feature and group valid masks differ")
            group_tensor = torch.from_numpy(group_values).to(device, non_blocking=True)
            with base._autocast(device):
                output = model(tensor, valid_mask=feature_valid_tensor)
                robust_total, _output_loss, robust_receipt = _group_robust_loss(
                    base,
                    output,
                    event,
                    boundary,
                    kinds,
                    valid,
                    group_tensor,
                    candidate=candidate,
                    positive_weight=positive_weight,
                    group_count=group_count,
                    strength=float(config["group_objective"]["strength"]),
                )
                loss = robust_total / accumulation
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite robust training loss")
            loss.backward()
            batch_weight = len(batch_windows)
            sums["total_loss"] += float(robust_total.detach().float().cpu()) * batch_weight
            for name in ("pooled_stage_bce", "worst_group_stage_bce"):
                sums[name] += float(robust_receipt[name].detach().float().cpu()) * batch_weight
            observed += batch_weight
            at_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
            if at_boundary:
                lr = base._lr_at_step(
                    global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    maximum_lr=float(candidate["training"]["learning_rate"]),
                    minimum_lr=3e-6,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(candidate["training"]["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("non-finite gradient norm")
                grad_norms.append(float(norm.detach().float().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        if not grad_norms:
            raise ContractError("training epoch performed no optimizer step")
        history.append(
            {
                "epoch": epoch,
                **{name: value / max(1, observed) for name, value in sums.items()},
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps_epoch": global_step - steps_before,
                "optimizer_steps_cumulative": global_step,
                "grad_norm_mean": sum(grad_norms) / len(grad_norms),
                "grad_norm_max": max(grad_norms),
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
                "nonfinite_count": 0,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config["training_contract"]["epochs"]):
            base._atomic_json(history_path, history, replace=True)
            base._atomic_json(
                ARTIFACT_DIR / "progress.json",
                {
                    "experiment_id": EXPERIMENT_ID,
                    "stage": phase,
                    "seed": seed,
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - run_started,
                    "performance_metrics_exposed": False,
                },
                replace=True,
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
        "seed": seed,
        "epochs": int(config["training_contract"]["epochs"]),
        "optimizer_steps": global_step,
        "parameter_count": int(model.trainable_parameter_count),
        "group_receipt": group_receipt,
        "history_artifact": {
            "path": history_path.relative_to(ARTIFACT_DIR).as_posix(),
            "bytes": history_path.stat().st_size,
            "sha256": _sha256(history_path),
        },
        "checkpoint_persisted": False,
        **base._cuda_peak_memory_receipt(torch, device),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return blind, receipt


def _ensemble_phase(
    base: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    device: Any,
    run_started: float,
) -> tuple[Any, list[dict[str, Any]]]:
    np, _pd, _torch, _model, _data = base._load_scientific()
    row = np.zeros(holdout.surface.rows, dtype=np.float32)
    boundary = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
    kinds = np.zeros((holdout.surface.rows, len(base.TYPE_NAMES)), dtype=np.float32)
    receipts: list[dict[str, Any]] = []
    seeds = config["training_contract"]["seeds"]
    for seed in seeds:
        blind, receipt = _fit_one(
            base,
            training,
            holdout,
            source_config=source_config,
            config=config,
            seed=int(seed),
            phase=phase,
            device=device,
            run_started=run_started,
        )
        row += blind.row_probability
        boundary += blind.boundary_probability
        kinds += blind.type_probability
        receipts.append(receipt)
    count = float(len(seeds))
    return base.PredictionBundle(row / count, boundary / count, kinds / count), receipts


def execute(*, expected_runner_sha256: str) -> dict[str, Any]:
    if expected_runner_sha256.casefold() != _sha256(Path(__file__)):
        raise ContractError("expected runner hash differs")
    if ARTIFACT_DIR.exists() or ATTEMPT_LOCK.exists():
        raise FileExistsError("one-shot namespace already exists")
    preflight = check_only()
    if not preflight["cuda_available"]:
        raise ContractError("Group-DRO execution requires CUDA")
    config = _config()
    base = SOBOL._load_base(root=ROOT)
    source_config = base._canonical_config()
    np, _pd, torch, _model, _data = base._load_scientific()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": preflight["runner_sha256"],
        "config_sha256": preflight["config_sha256"],
        "one_shot": True,
    }
    base._exclusive_json(ATTEMPT_LOCK, lock)
    ARTIFACT_DIR.mkdir(parents=False, exist_ok=False)
    started = datetime.now(UTC)
    run_started = time.perf_counter()
    terminal = ARTIFACT_DIR / "terminal_result.json"
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": preflight["config_sha256"],
        "runner_sha256": preflight["runner_sha256"],
        "helper_sha256": preflight["helper_sha256"],
        "fit_budget": 9,
    }
    commitment_sha = hashlib.sha256(
        json.dumps(commitment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        base._atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
        base._atomic_json(ARTIFACT_DIR / "commitment.json", commitment)
        device = torch.device("cuda")
        surfaces = base.load_blind_surfaces(source_config, root=ROOT)
        controls = SOBOL._control_candidates(base, json.loads((ROOT / config["source_pins"]["sobol_config"]["path"]).read_text(encoding="utf-8")), surfaces, root=ROOT)
        q2_encoder, q2_train, q2, q2_split = base._prepare_phase_surfaces(
            surfaces, source_config, "q2", root=ROOT
        )
        base._atomic_json(ARTIFACT_DIR / "q2_split.json", q2_split)
        base._atomic_json(ARTIFACT_DIR / "q2_encoder.json", base._encoder_receipt(q2_encoder))
        bundle, q2_receipts = _ensemble_phase(
            base,
            q2_train,
            q2,
            source_config=source_config,
            config=config,
            phase="q2",
            device=device,
            run_started=run_started,
        )
        decode_config = {"training_contract": {"threshold_grid": config["training_contract"]["threshold_grid"]}}
        proposals, candidates = SOBOL._decode_grid(base, bundle, q2, decode_config)
        thresholds = np.asarray(config["training_contract"]["threshold_grid"], dtype=np.float64)
        q2_path = SOBOL._seal_npz(
            base,
            ARTIFACT_DIR,
            name="q2_group_dro_blind",
            arrays={
                "thresholds": thresholds,
                "row_probability": bundle.row_probability,
                "boundary_probability": bundle.boundary_probability,
                "type_probability": bundle.type_probability,
                "proposal": proposals,
                "candidate": candidates,
            },
            config_sha256=preflight["config_sha256"],
            design_sha256=commitment_sha,
            key_sha256=surfaces.membership_sha256["2025_q2"],
            fit_receipts=q2_receipts,
        )
        q2_truth = SOBOL._load_truth(
            base,
            source_config,
            q2.surface,
            [q2_path],
            fold="2025_q2",
            config_sha256=preflight["config_sha256"],
            design_sha256=commitment_sha,
            root=ROOT,
        )
        records: list[dict[str, Any]] = []
        for index, threshold in enumerate(thresholds.tolist()):
            metrics = SOBOL._score_candidate(
                base, q2_truth, q2.surface.keys, candidates[index], controls["q2"]
            )
            concentration = changed_row_concentration(
                q2.surface.keys["station"], candidates[index], controls["q2"]
            )
            records.append(
                {"threshold": threshold, "threshold_index": index, "metrics": metrics, "changed_row_concentration": concentration}
            )
        winner = max(
            records,
            key=lambda row: (
                row["metrics"]["minimum_monthly_delta_f1"],
                row["metrics"]["pooled_delta_f1"],
                row["threshold"],
            ),
        )
        removed = int(
            np.sum((q2.surface.anchor == 1) & (candidates[winner["threshold_index"]] == 0))
        )
        q2_checks = {
            "all_monthly_delta_f1_strictly_positive": all(
                row["delta_f1"] > 0.0 for row in winner["metrics"]["monthly"].values()
            ),
            "pooled_delta_f1_strictly_positive": winner["metrics"]["pooled_delta_f1"] > 0.0,
            "anchor_positive_removed_rows_eq_0": removed == 0,
            "maximum_station_changed_row_share_lte_0_8": winner["changed_row_concentration"]["maximum_station_share"] <= 0.8,
        }
        q2_result = {"records": records, "winner": winner, "anchor_positive_removed_rows": removed, "checks": q2_checks, "pass": all(q2_checks.values())}
        base._atomic_json(ARTIFACT_DIR / "q2_gate.json", q2_result)
        if not q2_result["pass"]:
            result = {
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_Q2",
                "elapsed_seconds": time.perf_counter() - run_started,
                "historical_fit_count": 3,
                "q2": q2_result,
                "q3_q4_training_started": False,
                "official_interface_rows_read": 0,
                "csv_created": False,
                "upload_performed": False,
            }
            base._atomic_json(ARTIFACT_DIR / "aggregate.json", result)
            base._atomic_json(terminal, result)
            return result
        selected_threshold = float(winner["threshold"])
        receipt_paths: dict[str, Path] = {}
        holds: dict[str, Any] = {}
        selected_candidates: dict[str, Any] = {}
        for phase in ("q3", "q4"):
            encoder, training, holdout, split = base._prepare_phase_surfaces(
                surfaces, source_config, phase, root=ROOT
            )
            base._atomic_json(ARTIFACT_DIR / f"{phase}_split.json", split)
            base._atomic_json(ARTIFACT_DIR / f"{phase}_encoder.json", base._encoder_receipt(encoder))
            phase_bundle, receipts = _ensemble_phase(
                base,
                training,
                holdout,
                source_config=source_config,
                config=config,
                phase=phase,
                device=device,
                run_started=run_started,
            )
            phase_proposals, phase_candidates = SOBOL._decode_grid(
                base, phase_bundle, holdout, decode_config
            )
            threshold_index = int(np.flatnonzero(np.isclose(thresholds, selected_threshold))[0])
            path = SOBOL._seal_npz(
                base,
                ARTIFACT_DIR,
                name=f"{phase}_group_dro_blind",
                arrays={
                    "row_probability": phase_bundle.row_probability,
                    "boundary_probability": phase_bundle.boundary_probability,
                    "type_probability": phase_bundle.type_probability,
                    "proposal": phase_proposals[threshold_index],
                    "candidate": phase_candidates[threshold_index],
                },
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
                key_sha256=surfaces.membership_sha256[{"q3": "2025_q3", "q4": "2025_q4"}[phase]],
                fit_receipts=receipts,
            )
            receipt_paths[phase] = path
            holds[phase] = holdout
            selected_candidates[phase] = phase_candidates[threshold_index]
        for path in receipt_paths.values():
            SOBOL._verify_receipt(
                path,
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
            )
        truths = {
            phase: SOBOL._load_truth(
                base,
                source_config,
                holds[phase].surface,
                [receipt_paths[phase]],
                fold={"q3": "2025_q3", "q4": "2025_q4"}[phase],
                config_sha256=preflight["config_sha256"],
                design_sha256=commitment_sha,
                root=ROOT,
            )
            for phase in ("q3", "q4")
        }
        confirmation = SOBOL._evaluate_confirmatory(
            base, truths, holds, selected_candidates, controls
        )
        stations = np.concatenate([holds[phase].surface.keys["station"].astype(str).to_numpy() for phase in ("q3", "q4")])
        candidate_all = np.concatenate([selected_candidates[phase] for phase in ("q3", "q4")])
        control_all = np.concatenate([controls[phase] for phase in ("q3", "q4")])
        concentration = changed_row_concentration(stations, candidate_all, control_all)
        confirmation_checks = {
            **confirmation["checks"],
            "maximum_station_changed_row_share_lte_0_8": concentration["maximum_station_share"] <= 0.8,
        }
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "PASS_RETROSPECTIVE_ONLY" if all(confirmation_checks.values()) else "NO_GO_CONFIRMATION",
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - run_started,
            "historical_fit_count": 9,
            "selected_threshold": selected_threshold,
            "q2": q2_result,
            "confirmation": confirmation,
            "confirmation_changed_row_concentration": concentration,
            "confirmation_checks": confirmation_checks,
            "official_interface_rows_read": 0,
            "csv_created": False,
            "upload_performed": False,
            "result_based_rerun_authorized": False,
        }
        base._atomic_json(ARTIFACT_DIR / "aggregate.json", result)
        base._atomic_json(terminal, result)
        return result
    except BaseException as error:
        if ARTIFACT_DIR.exists() and not terminal.exists():
            base._atomic_json(
                terminal,
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_EXECUTION_NO_RETRY_AUTHORIZED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "official_interface_rows_read": 0,
                    "csv_created": False,
                    "upload_performed": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check-only", "execute"), required=True)
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args()
    if args.mode == "check-only":
        print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True))
        return 0
    if not args.expected_runner_sha256:
        parser.error("--expected-runner-sha256 is required for execute")
    print(
        json.dumps(
            execute(expected_runner_sha256=args.expected_runner_sha256),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

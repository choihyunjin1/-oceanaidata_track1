"""Clean P2 restoration comparison; only organizer observations.csv is read.

Reuse raw v23/v52 DeepSets and the train-fitted OAS kernel, never legacy
submission predictions, bin17, or official-score-derived coefficients.
All row-level outputs stay in the ignored experiment artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import psutil
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts/final_submission_20260905/P2"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from p2_pipeline import (  # noqa: E402
    MaskedThirdCentralMomentProfileVerticalDeepSet,
    build_arrays,
    domain_balanced_weights,
    observed_temperature_gradient_penalty,
    predict_model,
)
from run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 import (  # noqa: E402
    VerticalDeepSet,
)

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel  # noqa: E402
from p2_restore.features import PUBLIC_LAYERS, TARGET_LAYERS, _common_features  # noqa: E402
from p2_restore.normalized_curvature_residual import compute_profile_scale  # noqa: E402
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    predict_forward_seasonal_oas,
)

EXPERIMENT = "p2_score_repair_20260905_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT
REPORT = ROOT / "reports" / EXPERIMENT
DEPENDENCIES = (
    "scripts/final_submission_20260905/P2/p2_pipeline.py",
    "scripts/run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12.py",
    "src/p2_restore/features.py",
    "src/p2_restore/normalized_curvature_residual.py",
    "src/p2_restore/depth_registered_cmfpca.py",
    "src/p2_restore/p2_alpha40_quasiperiodic_gp_residual_20260828_v1.py",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == EXPERIMENT
    assert config["maximum_deepset_fits"] == 24 and len(config["folds"]) == 3
    assert config["cpu_threads"] == 1 and config["loader_workers"] == 0
    assert config["feature_dependency_hours"] == 0 and config["purge_days"] >= 7
    assert not any(config[key] for key in ("official_access", "csv_written", "upload"))
    return config


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def restoration_masks(times: pd.Series | pd.DatetimeIndex, fold: dict, purge_days: int) -> tuple[np.ndarray, np.ndarray]:
    parsed = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    left, right = utc(fold["start"]), utc(fold["stop"])
    purge = pd.Timedelta(days=purge_days)
    train = np.asarray((parsed < left - purge) | (parsed >= right + purge))
    valid = np.asarray((parsed >= left) & (parsed < right))
    assert not np.any(train & valid)
    return train, valid


def nominal_baseline(frame: pd.DataFrame) -> np.ndarray:
    """One contract for train/query: nominal interpolation with endpoint clamp."""
    temperatures = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    depths = frame[[f"nominal_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    targets = frame["target_depth"].to_numpy(float)
    result = np.full(len(frame), np.nan)
    for index, target in enumerate(targets):
        keep = np.isfinite(temperatures[index]) & np.isfinite(depths[index])
        if keep.sum() < 2 or not np.isfinite(target):
            continue
        order = np.argsort(depths[index, keep], kind="stable")
        result[index] = np.interp(target, depths[index, keep][order], temperatures[index, keep][order])
    return result


def refresh_public(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    values = frame[[f"temp_{layer}" for layer in PUBLIC_LAYERS]].to_numpy(float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    minimum = np.min(np.where(finite, values, np.inf), axis=1)
    maximum = np.max(np.where(finite, values, -np.inf), axis=1)
    frame["public_temp_count"] = count
    frame["public_temp_range"] = np.where(count > 0, maximum - minimum, np.nan)
    frame["baseline"] = nominal_baseline(frame)
    # A truth-free placeholder is required by legacy array validation, not used
    # as a feature or training target. Real labels are maintained separately.
    frame["target"] = frame["baseline"]
    return frame


def public_frame(observations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    public = observations.loc[observations["layer"].isin(PUBLIC_LAYERS)].copy()
    times, common = _common_features(public)
    lookup = pd.DataFrame(common, index=pd.DatetimeIndex(times))
    metadata = observations.loc[observations["layer"].isin(TARGET_LAYERS), ["time", "layer", "nominal_depth", "depth"]].copy()
    metadata = metadata.rename(columns={"nominal_depth": "target_depth", "depth": "target_actual_depth"})
    labels = observations.loc[observations["layer"].isin(TARGET_LAYERS), "temp"].to_numpy(float)
    frame = metadata.join(lookup, on="time", validate="many_to_one").reset_index(drop=True)
    frame["station"] = "S-ORS"
    frame = refresh_public(frame)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).astype(str)
    assert not {"temp_2", "temp_3", "temp_4", "psal_2", "psal_3", "psal_4"}.intersection(frame)
    return frame, labels


def arrays(frame: pd.DataFrame, actualdepth: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens, mask, context = build_arrays(frame)
    if actualdepth:
        depth = frame["target_actual_depth"].to_numpy(float)
        present = np.isfinite(depth) & (depth > 0)
        effective = np.where(present, depth, frame["target_depth"].to_numpy(float))
        context = np.column_stack((context, effective / 50.0, present)).astype(np.float32)
    if not all(np.isfinite(value).all() for value in (tokens, mask, context)):
        raise ValueError("nonfinite feature array")
    return tokens, mask, context


def block_selection(times: pd.Series, config: dict) -> np.ndarray:
    """Fixed calendar blocks, independent of labels and measured performance."""
    parsed = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    unique = parsed.unique().sort_values()
    rng = np.random.default_rng(config["seed"])
    selected = np.zeros(len(unique), dtype=bool)
    target = int(np.ceil(config["coverage"] * len(unique)))
    for _ in range(10000):
        if selected.sum() >= target:
            break
        start = unique[int(rng.integers(len(unique)))]
        stop = start + pd.Timedelta(days=int(rng.choice(config["length_days"])))
        selected |= np.asarray((unique >= start) & (unique < stop))
    return np.asarray(parsed.isin(unique[selected]))


def training_arrays(frame: pd.DataFrame, truth: np.ndarray, arm: str, config: dict) -> tuple[tuple[np.ndarray, ...], dict]:
    weights, weight_receipt = domain_balanced_weights(frame["layer"].to_numpy(), pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul"))
    origin_count = len(frame)
    duplicated = 0
    if arm.endswith("blockmask"):
        mask_rows = block_selection(frame["time"], config["blockmask"])
        masked = frame.copy()
        masked.loc[mask_rows, ["temp_5", "psal_5"]] = np.nan
        masked = refresh_public(masked)
        eligible = mask_rows & (masked["public_temp_count"].to_numpy() >= 2) & np.isfinite(masked["baseline"].to_numpy())
        duplicated = int(eligible.sum())
        augmentation_weights = weights[eligible] * 0.5
        weights[eligible] *= 0.5
        weights = np.concatenate((weights, augmentation_weights))
        frame = pd.concat((frame, masked.loc[eligible]), ignore_index=True)
        truth = np.concatenate((truth, truth[eligible]))
    feature_arrays = arrays(frame, arm.endswith("actualdepth"))
    target = ((truth - frame["baseline"].to_numpy(float)) / compute_profile_scale(frame)).astype(np.float32)
    assert np.isfinite(target).all() and np.isclose(weights.sum(), origin_count, rtol=1e-5)
    return (*feature_arrays, target, weights), {"original_rows": origin_count, "augmented_rows": duplicated, "training_weight_sum": float(weights.sum()), "domain_weights": weight_receipt}


def make_model(arm: str, context_features: int) -> torch.nn.Module:
    cls = MaskedThirdCentralMomentProfileVerticalDeepSet if arm.startswith("v52") else VerticalDeepSet
    return cls(8, context_features, hidden=32)


def fit_model(data: tuple[np.ndarray, ...], arm: str, seed: int, config: dict, progress) -> tuple[torch.nn.Module, dict]:
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")
    model = make_model(arm, data[2].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Whole numeric arrays fit comfortably in VRAM; one worker, no CPU loaders.
    tensors = tuple(torch.from_numpy(value).to(device) for value in data)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(data[0]), generator=generator).to(device)
        numerator, denominator = 0.0, 0.0
        for offset in range(0, len(order), config["batch_size"]):
            ids = order[offset:offset + config["batch_size"]]
            tokens = tensors[0][ids].detach().clone().requires_grad_(True)
            batch_mask, context, target, weights = [value[ids] for value in tensors[1:]]
            optimizer.zero_grad(set_to_none=True)
            estimate = model(tokens, batch_mask, context)
            raw_loss = F.smooth_l1_loss(estimate, target, beta=1.0, reduction="none")
            loss = (raw_loss * weights).sum() / weights.sum().clamp_min(1e-12)
            penalty = observed_temperature_gradient_penalty(raw_loss, tokens, batch_mask, weights)
            objective = loss + config["gradient_coefficient"] * penalty
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite training objective")
            objective.backward()
            optimizer.step()
            numerator += float((raw_loss.detach() * weights).sum().cpu())
            denominator += float(weights.sum().cpu())
        losses.append(numerator / denominator)
        if epoch % 10 == 0 or epoch + 1 == config["epochs"]:
            progress(epoch + 1, time.monotonic() - started)
    model = model.cpu().eval()
    del tensors, optimizer
    torch.cuda.empty_cache()
    return model, {"seed": seed, "epochs": config["epochs"], "parameters": sum(p.numel() for p in model.parameters()), "loss_first": losses[0], "loss_last": losses[-1], "runtime_seconds": time.monotonic() - started, "device": "cuda", "cpu_threads": 1}


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metric shape/finite contract")
    error = prediction - truth
    return {"n": len(error), "sse": float(error @ error), "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}


def missing_runs(mask: np.ndarray) -> dict:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {"missing_count": int(mask.sum()), "run_count": len(lengths), "max_run_steps": int(lengths.max()) if len(lengths) else 0, "runs_ge_144_steps": int((lengths >= 144).sum())}


def execute() -> dict:
    config = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("exclusive GPU contract unavailable; no automatic CPU fallback")
    source_dir = os.environ.get("P2_DATA_DIR")
    if not source_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    source = Path(source_dir).resolve() / "observations.csv"
    if file_hash(source) != config["source_sha256"]:
        raise RuntimeError("observations source SHA mismatch")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    if (ARTIFACT / "result.json").exists():
        raise FileExistsError("terminal artifact already exists")
    dependencies = {relative: file_hash(ROOT / relative) for relative in DEPENDENCIES}
    manifest = {"experiment_id": EXPERIMENT, "runner_sha256": file_hash(Path(__file__)), "config_sha256": file_hash(CONFIG), "source_sha256": config["source_sha256"], "source_allowed": ["observations.csv"], "dependencies": dependencies, "pid": os.getpid()}
    with (ARTIFACT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    atomic_json(REPORT / "manifest.json", manifest)
    started = time.monotonic()
    fits: list[dict] = []
    oas_receipts: list[dict] = []
    def progress(stage, **extra):
        payload = {"experiment_id": EXPERIMENT, "status": "RUNNING", "stage": stage, "pid": os.getpid(), "deepset_fits_completed": len(fits), "wall_seconds": time.monotonic() - started, "rss_gb": psutil.Process().memory_info().rss / 2**30, **extra}
        if payload["rss_gb"] > 12:
            raise MemoryError("12 GiB process RSS budget exceeded")
        atomic_json(ARTIFACT / "progress.json", payload)
    try:
        progress("load_train_observations")
        observations = pd.read_csv(source)
        if set(observations.columns) != {"station", "year", "time", "layer", "depth", "nominal_depth", "temp", "psal"}:
            raise ValueError("unexpected observations schema; inspect contract without values")
        observations["time"] = pd.to_datetime(observations["time"], utc=True)
        if observations.duplicated(["station", "time", "layer"]).any():
            raise ValueError("duplicate source key")
        observations = observations.loc[(observations["time"] >= utc(config["train_start"])) & (observations["time"] < utc(config["train_stop"]))].copy()
        frame, labels = public_frame(observations)
        eligible = np.isfinite(labels) & np.isfinite(frame["baseline"]) & (frame["public_temp_count"].to_numpy() >= 2)
        frame, labels = frame.loc[eligible].reset_index(drop=True), labels[eligible]
        parsed = pd.to_datetime(frame["time"], utc=True)
        keys = frame["time"].astype(str) + "|" + frame["layer"].astype(str)
        assert not keys.duplicated().any()
        fold_ids = np.full(len(frame), "", dtype="U20")
        folds = []
        for fold in config["folds"]:
            train, valid = restoration_masks(parsed, fold, config["purge_days"])
            if not train.any() or not valid.any():
                raise ValueError("empty fold")
            fold_ids[valid] = fold["id"]
            folds.append((fold, train, valid))
        validation = fold_ids != ""
        eval_frame = frame.loc[validation].reset_index(drop=True)
        truth = labels[validation]
        eval_fold = fold_ids[validation]
        predictions: dict[str, np.ndarray] = {"nominal": eval_frame["baseline"].to_numpy(float)}
        t1 = eval_frame["temp_1"].to_numpy(float)
        predictions["T1_else_nominal"] = np.where(np.isfinite(t1), t1, predictions["nominal"])
        predictions["oas"] = np.full(len(truth), np.nan)
        panel, _, target_columns = build_layer_identity_panel(observations)
        panel_index = panel.index.as_unit("ns")
        support = {"qc_columns_present": False, "selection_proxy": "finite target temp and >=2 public temperature tokens; not hidden QC", "source_rows_in_date_range": len(observations), "eligible_target_rows": len(frame), "evaluation_rows": len(truth), "target_actual_depth": {"finite_positive_fraction": float((np.isfinite(frame["target_actual_depth"]) & frame["target_actual_depth"].gt(0)).mean())}, "T5": missing_runs(~np.isfinite(panel["temp_5"].to_numpy())), "S5": missing_runs(~np.isfinite(panel["psal_5"].to_numpy())), "timestamp_step_seconds_counts": {str(k): int(v) for k, v in pd.Series(np.diff(panel_index.asi8) / 1e9).value_counts().items()}, "folds": []}
        for fold, train, valid in folds:
            left, right = utc(fold["start"]), utc(fold["stop"])
            isolated_panel = panel.copy()
            hidden = (panel_index >= left) & (panel_index < right)
            isolated_panel.loc[hidden, target_columns] = np.nan
            pred, receipts = predict_forward_seasonal_oas(isolated_panel, frame.loc[valid, ["time", "layer"]], train_stop=utc(config["train_stop"]), exclude_start=left - pd.Timedelta(days=config["purge_days"]), exclude_stop=right + pd.Timedelta(days=config["purge_days"]), **config["oas"])
            predictions["oas"][eval_fold == fold["id"]] = pred
            oas_receipts.append({"fold": fold["id"], "fit_count": len(receipts), "fits": receipts})
            support["folds"].append({"id": fold["id"], "calendar_days": int((right - left).days), "train_rows": int(train.sum()), "validation_rows": int(valid.sum()), "train_before": int((train & (parsed < left).to_numpy()).sum()), "train_after": int((train & (parsed >= right).to_numpy()).sum()), "target_temp_psal_masked": True, "feature_temporal_dependency_hours": 0, "purge_days": config["purge_days"]})
        atomic_json(REPORT / "data-contract.json", support)
        np.savez_compressed(ARTIFACT / "evaluation_keys.npz", key=keys[validation].to_numpy(str), fold=eval_fold, truth=truth, layer=eval_frame["layer"].to_numpy(), time=eval_frame["time"].to_numpy(str))
        def run_arm(arm: str, seed: int) -> None:
            output = np.full(len(truth), np.nan)
            for fold, train, valid in folds:
                if len(fits) >= config["maximum_deepset_fits"]:
                    raise RuntimeError("DeepSets fit cap")
                fit_id = f"{arm}_{seed}_{fold['id']}"
                progress("prepare_fit", fit_id=fit_id)
                data, training_receipt = training_arrays(frame.loc[train].reset_index(drop=True), labels[train], arm, config)
                model, receipt = fit_model(data, arm, seed, config, lambda epoch, elapsed, current_id=fit_id: progress("training", fit_id=current_id, epoch=epoch, fit_seconds=elapsed))
                query = frame.loc[valid].reset_index(drop=True)
                query_arrays = arrays(query, arm.endswith("actualdepth"))
                normalized = predict_model(model, *query_arrays)
                prediction = query["baseline"].to_numpy(float) + normalized * compute_profile_scale(query)
                checkpoint = ARTIFACT / f"{fit_id}.pt"
                torch.save(model.cpu().state_dict(), checkpoint)
                replay = make_model(arm, query_arrays[2].shape[1])
                replay.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                replay_pred = predict_model(replay, *query_arrays)
                if not np.array_equal(normalized, replay_pred):
                    raise RuntimeError("saved model inference mismatch")
                output[eval_fold == fold["id"]] = prediction
                receipt.update({"fit_id": fit_id, "arm": arm, "fold": fold["id"], "training": training_receipt, "model_sha256": file_hash(checkpoint), "saved_model_inference_exact": True, "metrics": metrics(labels[valid], prediction)})
                fits.append(receipt)
                atomic_json(ARTIFACT / "fit-receipts.json", fits)
                np.savez_compressed(ARTIFACT / f"{fit_id}.npz", key=keys[valid].to_numpy(str), truth=labels[valid], prediction=prediction, raw_normalized=normalized)
                del model, replay, data, query_arrays
                torch.cuda.empty_cache()
                print(json.dumps({"fit_completed": fit_id, "total": len(fits), "seconds": receipt["runtime_seconds"]}), flush=True)
            predictions[f"{arm}_seed{seed}"] = output
            atomic_json(ARTIFACT / "screen-progress.json", {key: metrics(truth, value) for key, value in predictions.items()})
        initial_seed = config["seeds"][0]
        for arm in ("v23", "v52"):
            run_arm(arm, initial_seed)
        control = min(("v23", "v52"), key=lambda arm: metrics(truth, predictions[f"{arm}_seed{initial_seed}"])["rmse"])
        atomic_json(REPORT / "screen1.json", {"control": control, "metrics": {key: metrics(truth, value) for key, value in predictions.items()}, "fit_count": len(fits)})
        for arm in (f"{control}_blockmask", f"{control}_actualdepth"):
            run_arm(arm, initial_seed)
        candidates = [control, f"{control}_blockmask", f"{control}_actualdepth"]
        winner = min(candidates, key=lambda arm: metrics(truth, predictions[f"{arm}_seed{initial_seed}"])["rmse"])
        confirmed = [control] if winner == control else [control, winner]
        # If no ablation beats control, do not spend duplicate confirmation fits.
        if winner != control:
            for seed in config["seeds"][1:]:
                for arm in confirmed:
                    run_arm(arm, seed)
        for arm in confirmed:
            members = [value for key, value in predictions.items() if key.startswith(f"{arm}_seed")]
            predictions[f"{arm}_mean"] = np.mean(members, axis=0)
        reference = predictions[f"{control}_mean"]
        chosen = predictions[f"{winner}_mean"]
        # This fixed average is a declared secondary diagnostic, not a fitted weight.
        predictions["chosen_oas_fixed_half"] = 0.5 * chosen + 0.5 * predictions["oas"]
        all_metrics = {key: metrics(truth, value) for key, value in predictions.items()}
        chosen_name = f"{winner}_mean"
        np.savez_compressed(ARTIFACT / "raw_oof.npz", key=keys[validation].to_numpy(str), fold=eval_fold, truth=truth, **predictions)
        np.savez_compressed(ARTIFACT / "qa_oof.npz", key=keys[validation].to_numpy(str), fold=eval_fold, truth=truth, reference=reference, prediction=chosen)
        strata = {}
        local = pd.to_datetime(eval_frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
        for kind, groups in {"fold": eval_fold, "layer": eval_frame["layer"].astype(str).to_numpy(), "month": local.dt.month.astype(str).to_numpy(), "T5_present": np.isfinite(eval_frame["temp_5"]).astype(str)}.items():
            strata[kind] = {str(group): {"reference": metrics(truth[groups == group], reference[groups == group]), "chosen": metrics(truth[groups == group], chosen[groups == group])} for group in np.unique(groups)}
        result = {"experiment_id": EXPERIMENT, "status": "COMPLETE_EXPLORATORY_INTERNAL_ONLY", "official_submission_ready": False, "control": control, "winner_screen_seed": winner, "chosen_prediction": chosen_name, "deepset_fit_count": len(fits), "oas_fit_count": sum(item["fit_count"] for item in oas_receipts), "calibration_fit_count": 0, "fullfit_count": 0, "metrics": all_metrics, "strata": strata, "rmse_delta_chosen_minus_control": metrics(truth, chosen)["rmse"] - metrics(truth, reference)["rmse"], "error_correlation_chosen_oas": float(np.corrcoef(chosen - truth, predictions["oas"] - truth)[0, 1]), "runtime_seconds": time.monotonic() - started, "official_access_rows": 0, "csv_written": 0, "upload": 0, "access_evidence": "runner only pd.read_csv at allowlisted source; no official loader called; no OS-wide audit claim", "limitations": ["Repeated historical folds, exploratory selection; not fresh confirmation", "Finite-target/public-support proxy, hidden QC unavailable", "Layer-ID OAS retains legacy coordinate representation", "November-December final fold has no post-window data; other folds use both sides", "Raw standalone models differ from historical bin17+clip blend; no historical score reproduction claim"], "manifest": manifest, "oas_receipts": oas_receipts, "fit_receipts": fits, "artifact_sha256": {name: file_hash(ARTIFACT / name) for name in ("raw_oof.npz", "qa_oof.npz", "evaluation_keys.npz")}}
        atomic_json(ARTIFACT / "result.json", result)
        atomic_json(REPORT / "result.json", result)
        atomic_json(ARTIFACT / "progress.json", {"status": "COMPLETE", "deepset_fits_completed": len(fits), "runtime_seconds": result["runtime_seconds"]})
        return result
    except Exception as exc:
        failure = {"experiment_id": EXPERIMENT, "status": "TERMINAL_TECHNICAL_FAILURE", "exception_type": type(exc).__name__, "error": str(exc), "fits_completed": len(fits), "runtime_seconds": time.monotonic() - started, "automatic_restart_allowed": False}
        atomic_json(ARTIFACT / "failure.json", failure)
        atomic_json(REPORT / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT, "config_valid": bool(load_config()), "execute": False}))
        return
    with threadpool_limits(limits=1):
        result = execute()
    print(json.dumps({"status": result["status"], "deepset_fit_count": result["deepset_fit_count"]}), flush=True)


if __name__ == "__main__":
    main()

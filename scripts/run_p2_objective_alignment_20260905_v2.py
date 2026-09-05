"""Sealed P2 loss/weight factorial; distributed observations and clean C only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_score_repair_20260905_v1 as base  # noqa: E402

ID = "p2_objective_alignment_20260905_v2"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
ARTIFACT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
SEAL = REPORT / "preregistration-seal.json"
OLD = ROOT / "artifacts/p2_score_repair_20260905_v1"
DEPENDENCIES = (
    "scripts/run_p2_score_repair_20260905_v1.py",
    "configs/experiments/p2_score_repair_20260905_v1.json",
    *base.DEPENDENCIES,
)


def load_config() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID
    assert cfg["screen_arms"] == ["M", "R", "MR"]
    assert cfg["maximum_new_historical_fits"] == 15
    assert cfg["seeds"] == [20260901, 20260902, 20260903]
    assert cfg["cpu_threads"] == 1 and cfg["loader_workers"] == 0
    assert cfg["feature_dependency_hours"] == 0 and cfg["purge_days"] == 7
    assert cfg["primary_fold"] == "2024_sep_oct"
    assert cfg["epochs"] == 60 and cfg["gradient_coefficient"] == 0.01
    assert cfg["penalty_contract"].endswith("fixed_domain_weights_for_all_arms")
    return cfg


def fingerprints() -> dict:
    return {
        "runner": base.file_hash(Path(__file__)),
        "config": base.file_hash(CONFIG),
        "dependencies": {p: base.file_hash(ROOT / p) for p in DEPENDENCIES},
        "old_result": base.file_hash(ROOT / "reports/p2_score_repair_20260905_v1/result.json"),
        "old_raw": base.file_hash(OLD / "raw_oof.npz"),
    }


def seal() -> dict:
    load_config()
    value = {"experiment_id": ID, "sealed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hashes": fingerprints(), "new_historical_fit_cap": 15, "source_allowed": ["observations.csv"]}
    REPORT.mkdir(parents=True, exist_ok=True)
    with SEAL.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    return value


def make_training_arrays(frame: pd.DataFrame, truth: np.ndarray, cfg: dict) -> tuple[tuple[np.ndarray, ...], dict]:
    old_data, receipt = base.training_arrays(frame, truth, "v23_blockmask", cfg)
    chosen = base.block_selection(frame["time"], cfg["blockmask"])
    altered = frame.copy()
    altered.loc[chosen, ["temp_5", "psal_5"]] = np.nan
    altered = base.refresh_public(altered)
    eligible = chosen & (altered["public_temp_count"].to_numpy() >= 2) & np.isfinite(altered["baseline"])
    expanded = pd.concat((frame, altered.loc[eligible]), ignore_index=True)
    scale = base.compute_profile_scale(expanded).astype(np.float32)
    equal = np.ones(len(frame), dtype=np.float32)
    equal[eligible] = 0.5
    equal = np.concatenate((equal, np.full(int(eligible.sum()), 0.5, dtype=np.float32)))
    assert len(scale) == len(old_data[0]) and equal.sum() == len(frame)
    # Order: tokens, mask, context, normalized target, immutable domain weights,
    # target-free temperature scale, separately preserved uniform data weights.
    return (*old_data, scale, equal), receipt


def objective_terms(estimate, target, scale, domain, equal, tokens, mask, arm):
    normalized_huber = F.smooth_l1_loss(estimate, target, beta=1.0, reduction="none")
    raw = (scale * (estimate - target)).square() if arm in ("M", "MR") else normalized_huber
    data_weights = equal if arm in ("R", "MR") else domain
    data_loss = (raw * data_weights).sum() / data_weights.sum().clamp_min(1e-12)
    penalty = base.observed_temperature_gradient_penalty(normalized_huber, tokens, mask, domain)
    return data_loss, penalty


def fit_model(data, arm, seed, cfg, progress):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = base.make_model("v23", data[2].shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    tensors = tuple(torch.from_numpy(value).cuda() for value in data)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.monotonic()
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(data[0]), generator=generator).cuda()
        sums = np.zeros(3)
        for offset in range(0, len(order), cfg["batch_size"]):
            ids = order[offset:offset + cfg["batch_size"]]
            tokens = tensors[0][ids].detach().clone().requires_grad_(True)
            mask, context, target, domain, scale, equal = [v[ids] for v in tensors[1:]]
            optimizer.zero_grad(set_to_none=True)
            data_loss, penalty = objective_terms(model(tokens, mask, context), target, scale, domain, equal, tokens, mask, arm)
            loss = data_loss + cfg["gradient_coefficient"] * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite objective")
            loss.backward()
            optimizer.step()
            sums += [float(data_loss.detach()), float(penalty.detach()), 1]
        history.append({"data_loss_batch_mean": sums[0] / sums[2], "penalty_batch_mean": sums[1] / sums[2]})
        if epoch % 10 == 0 or epoch + 1 == cfg["epochs"]:
            progress(epoch + 1)
    model = model.cpu().eval()
    del tensors, optimizer
    torch.cuda.empty_cache()
    return model, {"seed": seed, "arm": arm, "epochs": cfg["epochs"], "runtime_seconds": time.monotonic() - started, "objective_history": history}


def predict_absolute(model, frame):
    values = base.predict_model(model, *base.arrays(frame))
    return frame["baseline"].to_numpy(float) + values * base.compute_profile_scale(frame)


def stress_frame(frame, cfg):
    altered = frame.copy()
    local = pd.to_datetime(altered["time"], utc=True)
    selected = (local >= base.utc(cfg["stress"]["start"])) & (local < base.utc(cfg["stress"]["stop"]))
    altered.loc[selected, cfg["stress"]["masked_public_columns"]] = np.nan
    altered = base.refresh_public(altered)
    supported = np.isfinite(altered["baseline"]) & (altered["public_temp_count"].to_numpy() >= 2)
    return altered, np.asarray(supported), np.asarray(selected)


def load_data(cfg):
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    if base.file_hash(source) != cfg["source_sha256"]:
        raise RuntimeError("immutable distributed source hash mismatch")
    obs = pd.read_csv(source)
    if obs.duplicated(["station", "time", "layer"]).any():
        raise RuntimeError("duplicate distributed source keys")
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    obs = obs.loc[(obs.time >= base.utc(cfg["train_start"])) & (obs.time < base.utc(cfg["train_stop"]))].copy()
    frame, truth = base.public_frame(obs)
    keep = np.isfinite(truth) & np.isfinite(frame["baseline"]) & (frame["public_temp_count"].to_numpy() >= 2)
    frame, truth = frame.loc[keep].reset_index(drop=True), truth[keep]
    return frame, truth


def metrics_by_scope(truth, prediction, fold, primary):
    return {"primary": base.metrics(truth[primary], prediction[primary]), "pooled": base.metrics(truth, prediction), "fold": {f: base.metrics(truth[fold == f], prediction[fold == f]) for f in np.unique(fold)}}


def execute():
    cfg = load_config()
    torch.set_num_threads(1)
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    if sealed["hashes"] != fingerprints():
        raise RuntimeError("sealed execution hash drift")
    if not torch.cuda.is_available():
        raise RuntimeError("exclusive GPU unavailable")
    for name in ("01_data", "03_training", "04_models", "06_report"):
        (ARTIFACT / name).mkdir(parents=True, exist_ok=True)
    with (ARTIFACT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({**sealed, "pid": os.getpid()}, handle)
    started = time.monotonic()
    fits = []
    def progress(stage, **extra):
        base.atomic_json(ARTIFACT / "progress.json", {"status": "RUNNING", "pid": os.getpid(), "stage": stage, "new_fits": len(fits), "runtime_seconds": time.monotonic() - started, **extra})
    try:
        progress("source_and_control_reconciliation")
        frame, labels = load_data(cfg)
        parsed = pd.to_datetime(frame.time, utc=True)
        keys = (frame.time.astype(str) + "|" + frame.layer.astype(str)).to_numpy(str)
        folds = [(spec, *base.restoration_masks(parsed, spec, cfg["purge_days"])) for spec in cfg["folds"]]
        fold_ids = np.full(len(frame), "", dtype="U20")
        for spec, train, valid in folds:
            assert not np.any(train & valid)
            fold_ids[valid] = spec["id"]
        evaluated = fold_ids != ""
        query = frame.loc[evaluated].reset_index(drop=True)
        truth, eval_fold = labels[evaluated], fold_ids[evaluated]
        primary = eval_fold == cfg["primary_fold"]
        altered, stress_supported, stress_selected = stress_frame(query, cfg)
        old = np.load(OLD / "raw_oof.npz", allow_pickle=False)
        if base.file_hash(OLD / "raw_oof.npz") != cfg["control_raw_sha256"]:
            raise RuntimeError("control artifact hash mismatch")
        for name, values in (("key", keys[evaluated]), ("truth", truth), ("fold", eval_fold)):
            if not np.array_equal(old[name], values):
                raise RuntimeError(f"control {name} alignment mismatch")
        old_result = json.loads((ROOT / "reports/p2_score_repair_20260905_v1/result.json").read_text(encoding="utf-8"))
        old_receipts = {item["fit_id"]: item for item in old_result["fit_receipts"]}
        predictions, stress_predictions = {}, {}
        replay_receipts = []
        for seed in cfg["seeds"]:
            pred = np.full(len(truth), np.nan)
            stress_pred = np.full(len(truth), np.nan)
            for spec, _, valid in folds:
                fit_id = f"v23_blockmask_{seed}_{spec['id']}"
                checkpoint = OLD / f"{fit_id}.pt"
                if base.file_hash(checkpoint) != old_receipts[fit_id]["model_sha256"]:
                    raise RuntimeError("control checkpoint mismatch")
                model = base.make_model("v23", 11)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                subset = eval_fold == spec["id"]
                pred[subset] = predict_absolute(model, frame.loc[valid].reset_index(drop=True))
                allowed = subset & stress_supported
                stress_pred[allowed] = predict_absolute(model, altered.loc[allowed].reset_index(drop=True))
                replay_receipts.append({"fit_id": fit_id, "sha256": base.file_hash(checkpoint), "new_training_fits": 0})
            if not np.array_equal(pred, old[f"v23_blockmask_seed{seed}"]):
                raise RuntimeError("C fresh inference is not exact")
            predictions[f"C_seed{seed}"] = pred
            stress_predictions[f"C_seed{seed}"] = stress_pred
        base.atomic_json(REPORT / "control-reuse-qa.json", {"passed": True, "rows": len(truth), "same_key_truth_fold": True, "replay_exact": True, "checkpoint_receipts": replay_receipts})
        progress("C_reuse_exact")
        def run_arm(arm, seed):
            out, stress_out = np.full(len(truth), np.nan), np.full(len(truth), np.nan)
            for spec, train, valid in folds:
                if len(fits) >= cfg["maximum_new_historical_fits"]:
                    raise RuntimeError("sealed fit budget exceeded")
                fit_id = f"{arm}_{seed}_{spec['id']}"
                data, data_receipt = make_training_arrays(frame.loc[train].reset_index(drop=True), labels[train], cfg)
                progress("fit", fit_id=fit_id)
                model, receipt = fit_model(data, arm, seed, cfg, lambda epoch, current_id=fit_id: progress("training", fit_id=current_id, epoch=epoch))
                subset = eval_fold == spec["id"]
                pred = predict_absolute(model, frame.loc[valid].reset_index(drop=True))
                out[subset] = pred
                allowed = subset & stress_supported
                stress_out[allowed] = predict_absolute(model, altered.loc[allowed].reset_index(drop=True))
                checkpoint = ARTIFACT / "04_models" / f"{fit_id}.pt"
                torch.save(model.state_dict(), checkpoint)
                replay = base.make_model("v23", 11)
                replay.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                if not np.array_equal(pred, predict_absolute(replay, frame.loc[valid].reset_index(drop=True))):
                    raise RuntimeError("saved new model replay mismatch")
                receipt.update({"fit_id": fit_id, "fold": spec["id"], "training": data_receipt, "model_sha256": base.file_hash(checkpoint), "replay_exact": True})
                fits.append(receipt)
                base.atomic_json(ARTIFACT / "03_training/fit-receipts.json", fits)
                print(json.dumps({"completed": fit_id, "new_fits": len(fits), "seconds": receipt["runtime_seconds"]}), flush=True)
                del data, model, replay
            predictions[f"{arm}_seed{seed}"] = out
            stress_predictions[f"{arm}_seed{seed}"] = stress_out
        seed0 = cfg["seeds"][0]
        for arm in cfg["screen_arms"]:
            run_arm(arm, seed0)
        screen = {arm: metrics_by_scope(truth, predictions[f"{arm}_seed{seed0}"], eval_fold, primary) for arm in ("C", "M", "R", "MR")}
        order = {name: index for index, name in enumerate(("C", "M", "R", "MR"))}
        winner = min(screen, key=lambda arm: (screen[arm]["primary"]["rmse"], screen[arm]["pooled"]["rmse"], order[arm]))
        base.atomic_json(REPORT / "screen.json", {"winner": winner, "scores": screen, "new_fits": len(fits)})
        if winner != "C":
            for seed in cfg["seeds"][1:]:
                run_arm(winner, seed)
        for arm in ("C", winner):
            predictions[f"{arm}_mean"] = np.mean([predictions[f"{arm}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
            stress_predictions[f"{arm}_mean"] = np.mean([stress_predictions[f"{arm}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
        reference, chosen = predictions["C_mean"], predictions[f"{winner}_mean"]
        delta = base.metrics(truth[primary], chosen[primary])["rmse"] - base.metrics(truth[primary], reference[primary])["rmse"]
        metrics = {name: metrics_by_scope(truth, value, eval_fold, primary) for name, value in predictions.items()}
        scopes = {"fall_all_supported": primary & stress_supported, "fall_masked_supported": primary & stress_supported & stress_selected}
        stress_metrics = {scope: {name: base.metrics(truth[mask], value[mask]) for name, value in stress_predictions.items()} for scope, mask in scopes.items()}
        layers = query.layer.to_numpy(int)
        strata = {str(layer): {"reference": base.metrics(truth[layers == layer], reference[layers == layer]), "chosen": base.metrics(truth[layers == layer], chosen[layers == layer])} for layer in np.unique(layers)}
        arrays_path = ARTIFACT / "03_training/raw_oof.npz"
        np.savez_compressed(arrays_path, key=keys[evaluated], truth=truth, fold=eval_fold, layer=layers, time=query.time.to_numpy(str), stress_supported=stress_supported, stress_selected=stress_selected, **predictions, **{f"stress_{key}": value for key, value in stress_predictions.items()})
        result = {"experiment_id": ID, "status": "PRIMARY_IMPROVEMENT_INTERNAL_ONLY" if delta < 0 else "NO_PRIMARY_IMPROVEMENT_NEXT_P2_B", "winner_screen": winner, "chosen": f"{winner}_mean", "primary_delta_rmse_C": delta, "new_historical_fits": len(fits), "reused_C_models": 9, "calibration_fits": 0, "fulltrain_fits": 0, "metrics": metrics, "stress_metrics": stress_metrics, "layer_metrics": strata, "stress_support": {name: int(mask.sum()) for name, mask in scopes.items()}, "rows": len(truth), "training_eligible_rows": len(frame), "runtime_seconds": time.monotonic() - started, "official_access_rows": 0, "csv_written": 0, "upload": 0, "official_control_rmse_C": 0.455143, "official_control_points": 27.622418, "new_expected_official_points": None, "artifact_sha256": base.file_hash(arrays_path), "sealed_hashes": sealed["hashes"], "limitations": ["Historical development surfaces repeatedly exposed; not fresh confirmation", "Only one historical intact autumn is primary", "Absolute-C loss changes its scale relative to unchanged regularizer", "QC unavailable: finite target and >=2 public temperatures are proxy eligibility", "Stress is label-independent artificial outage; not proof of official missingness match"]}
        base.atomic_json(REPORT / "result.json", result)
        base.atomic_json(ARTIFACT / "06_report/result.json", result)
        base.atomic_json(ARTIFACT / "progress.json", {"status": "COMPLETE", "new_fits": len(fits), "runtime_seconds": result["runtime_seconds"]})
        return result
    except Exception as error:
        receipt = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(error).__name__, "message": str(error), "new_fits": len(fits), "runtime_seconds": time.monotonic() - started, "automatic_restart": False}
        base.atomic_json(REPORT / "failure.json", receipt)
        base.atomic_json(ARTIFACT / "failure.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.seal and args.execute:
        raise ValueError("seal must precede execution")
    if args.seal:
        print(json.dumps(seal(), indent=2))
    elif args.execute:
        with threadpool_limits(limits=1):
            result = execute()
        print(json.dumps({key: result[key] for key in ("status", "chosen", "primary_delta_rmse_C", "new_historical_fits", "runtime_seconds")}))
    else:
        print(json.dumps({"config_valid": bool(load_config()), "execute": False}))


if __name__ == "__main__":
    main()

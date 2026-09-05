"""P2-B fixed physical-slot L2 tree following the terminal objective screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_objective_alignment_20260905_v2 as a  # noqa: E402

base = a.base
ID = "p2_physical_profile_tree_20260905_v2"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
ARTIFACT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID
SEAL = REPORT / "preregistration-seal.json"
DEPENDENCIES = ("scripts/run_p2_objective_alignment_20260905_v2.py", "src/p2_restore/model.py", *a.DEPENDENCIES)


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID and cfg["maximum_new_historical_fits"] == 9
    assert cfg["primary_fold"] == "2024_sep_oct"
    assert cfg["feature_dependency_hours"] == 12 and cfg["purge_days"] == 7
    assert cfg["policies"] == ["C", "tree", "fixed_half"]
    assert cfg["model_parameters"]["n_jobs"] == 2
    return cfg


def fingerprint():
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG), "dependencies": {p: base.file_hash(ROOT / p) for p in DEPENDENCIES}, "A_terminal": base.file_hash(a.REPORT / "result.json"), "A_control_stress": base.file_hash(a.ARTIFACT / "03_training/raw_oof.npz"), "C_raw": base.file_hash(a.OLD / "raw_oof.npz")}


def physical_features(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """No target values; public rows mapped by nominal depth, not layer ID."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
    distinct = ~timestamps.duplicated()
    public = frame.loc[distinct].copy()
    public.index = timestamps[distinct]
    grid = np.asarray(cfg["depth_slots_m"], dtype=float)
    matrix = {}
    for slot in grid.astype(int):
        for value in ("temp", "psal", "depth", "nominal"):
            matrix[f"{value}_s{slot}"] = np.full(len(public), np.nan)
    occupied = np.zeros((len(public), len(grid)), dtype=bool)
    for layer in base.PUBLIC_LAYERS:
        nominal = public[f"nominal_{layer}"].to_numpy(float)
        valid = np.isfinite(nominal)
        assigned = np.argmin(np.abs(nominal[:, None] - grid[None, :]), axis=1)
        for index, slot in enumerate(grid.astype(int)):
            selected = valid & (assigned == index)
            if np.any(occupied[selected, index]):
                raise ValueError("two public sensors collide in a physical depth slot")
            occupied[selected, index] = True
            for value in ("temp", "psal", "depth", "nominal"):
                matrix[f"{value}_s{slot}"][selected] = public[f"{value}_{layer}"].to_numpy(float)[selected]
    for slot in grid.astype(int):
        for value in ("temp", "psal"):
            matrix[f"present_{value}_s{slot}"] = np.isfinite(matrix[f"{value}_s{slot}"]).astype(float)
        series = pd.Series(matrix[f"temp_s{slot}"], index=public.index)
        for lag in cfg["public_temperature_lags_hours"]:
            # Exact timestamp lookup: missing observations stay missing. Masking
            # happens before this function, including all T5 lag sources.
            neighbor = series.reindex(public.index + pd.Timedelta(hours=lag)).to_numpy(float)
            matrix[f"temp_s{slot}_delta_h{lag:+d}"] = neighbor - series.to_numpy(float)
    for left, right in zip(grid.astype(int)[:-1], grid.astype(int)[1:], strict=True):
        for value in ("temp", "psal"):
            matrix[f"{value}_diff_s{left}_s{right}"] = matrix[f"{value}_s{left}"] - matrix[f"{value}_s{right}"]
    shared = pd.DataFrame(matrix, index=public.index)
    result = shared.reindex(timestamps).reset_index(drop=True)
    for field in ("target_depth", "layer", "baseline", "public_temp_count"):
        result[field] = frame[field].to_numpy(float)
    endpoints = np.isfinite(frame.temp_1) & np.isfinite(frame.temp_5)
    raw_scale = np.where(endpoints, np.abs(frame.temp_1 - frame.temp_5), frame.public_temp_range)
    # Fully unsupported timestamps remain in the context timeline only; no
    # target from these rows is eligible for fitting or evaluation.
    result["profile_scale"] = np.maximum(np.nan_to_num(raw_scale, nan=0.5), 0.5)
    for field in ("doy_sin", "doy_cos", "hour_sin", "hour_cos", "m2_sin", "m2_cos"):
        result[field] = frame[field].to_numpy(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("infinite features")
    return result.astype(np.float32)


def masked_frame(frame, selected_times):
    altered = frame.copy()
    selected = pd.to_datetime(altered.time, utc=True).isin(selected_times)
    altered.loc[selected, ["temp_5", "psal_5"]] = np.nan
    return base.refresh_public(altered)


def make_training_data(frame, labels, train, original_features, cfg):
    train_rows = frame.loc[train].reset_index(drop=True)
    selected = base.block_selection(train_rows.time, cfg["blockmask"])
    times = pd.DatetimeIndex(pd.to_datetime(train_rows.loc[selected, "time"], utc=True)).unique()
    altered = masked_frame(frame, times)
    # Rows with <2 temperatures get temporary finite baseline only for feature
    # construction; they are never eligible for original/augmented training.
    supported = np.isfinite(altered.baseline) & (altered.public_temp_count >= 2)
    safe = altered.copy()
    safe.loc[~supported, "baseline"] = 0.0
    augmented_features = physical_features(safe, cfg)
    selected_global = pd.to_datetime(frame.time, utc=True).isin(times).to_numpy()
    eligible = train & selected_global & supported.to_numpy()
    first_weight = np.ones(int(train.sum()), dtype=float)
    first_weight[eligible[train]] = 0.5
    weight = np.concatenate((first_weight, np.full(int(eligible.sum()), 0.5)))
    x = pd.concat((original_features.loc[train], augmented_features.loc[eligible]), ignore_index=True)
    y = np.concatenate((labels[train] - frame.baseline.to_numpy()[train], labels[eligible] - altered.baseline.to_numpy()[eligible]))
    assert weight.sum() == train.sum() and np.isfinite(y).all()
    return x, y, weight, {"original_rows": int(train.sum()), "augmented_rows": int(eligible.sum()), "weight_sum": float(weight.sum())}


def load_all(cfg):
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    if base.file_hash(source) != cfg["source_sha256"]:
        raise RuntimeError("distributed source hash mismatch")
    obs = pd.read_csv(source)
    if obs.duplicated(["station", "time", "layer"]).any():
        raise RuntimeError("duplicate source keys")
    obs.time = pd.to_datetime(obs.time, utc=True)
    obs = obs.loc[(obs.time >= base.utc(cfg["train_start"])) & (obs.time < base.utc(cfg["train_stop"]))].copy()
    frame, labels = base.public_frame(obs)
    eligible = np.isfinite(labels) & np.isfinite(frame.baseline) & (frame.public_temp_count >= 2)
    # Features need the entire public timeline, including target-unlabelled rows.
    safe = frame.copy()
    unsupported = ~np.isfinite(safe.baseline)
    safe.loc[unsupported, "baseline"] = 0.0
    return frame, labels, eligible.to_numpy(), safe


def execute():
    cfg = load_config()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal["hashes"] != fingerprint():
        raise RuntimeError("P2-B seal drift")
    a_result = json.loads((a.REPORT / "result.json").read_text(encoding="utf-8"))
    if a_result["status"] != "NO_PRIMARY_IMPROVEMENT_NEXT_P2_B":
        raise RuntimeError("P2-B failure branch has not been reached")
    for directory in ("01_data", "03_training", "04_models", "06_report"):
        (ARTIFACT / directory).mkdir(parents=True, exist_ok=True)
    with (ARTIFACT / "ATTEMPT_LOCK.json").open("x", encoding="utf-8") as handle:
        json.dump({**seal, "pid": os.getpid()}, handle)
    started, fits = time.monotonic(), []
    def progress(stage, **extra):
        base.atomic_json(ARTIFACT / "progress.json", {"status": "RUNNING", "pid": os.getpid(), "stage": stage, "fits": len(fits), "runtime_seconds": time.monotonic() - started, **extra})
    try:
        progress("data_and_features")
        frame, labels, eligible, safe = load_all(cfg)
        x = physical_features(safe, cfg)
        folds, fold_ids = [], np.full(len(frame), "", dtype="U20")
        for spec in cfg["folds"]:
            train, valid = base.restoration_masks(frame.time, spec, cfg["purge_days"])
            train, valid = train & eligible, valid & eligible
            fold_ids[valid] = spec["id"]
            folds.append((spec, train, valid))
        evaluated = fold_ids != ""
        query = frame.loc[evaluated].reset_index(drop=True)
        truth, eval_fold = labels[evaluated], fold_ids[evaluated]
        primary = eval_fold == cfg["primary_fold"]
        keys = (query.time.astype(str) + "|" + query.layer.astype(str)).to_numpy(str)
        old_path = a.OLD / "raw_oof.npz"
        old = np.load(old_path, allow_pickle=False)
        if base.file_hash(old_path) != cfg["control_raw_sha256"] or not all(np.array_equal(old[name], value) for name, value in (("key", keys), ("truth", truth), ("fold", eval_fold))):
            raise RuntimeError("clean C population mismatch")
        control_stress = np.load(a.ARTIFACT / "03_training/raw_oof.npz", allow_pickle=False)
        stress_times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
        outage = (stress_times >= base.utc(cfg["stress"]["start"])) & (stress_times < base.utc(cfg["stress"]["stop"]))
        altered = masked_frame(frame, stress_times[outage].unique())
        supported = np.isfinite(altered.baseline) & (altered.public_temp_count >= 2)
        altered_safe = altered.copy()
        altered_safe.loc[~supported, "baseline"] = 0.0
        stress_x = physical_features(altered_safe, cfg)
        stress_supported = supported.to_numpy()[evaluated]
        predictions = {f"C_seed{seed}": old[f"v23_blockmask_seed{seed}"] for seed in cfg["seeds"]}
        stress_predictions = {f"C_seed{seed}": control_stress[f"stress_C_seed{seed}"] for seed in cfg["seeds"]}
        def run_seed(seed):
            out, stress_out = np.full(len(truth), np.nan), np.full(len(truth), np.nan)
            for spec, train, valid in folds:
                if len(fits) >= cfg["maximum_new_historical_fits"]:
                    raise RuntimeError("fit cap")
                fit_id = f"tree_{seed}_{spec['id']}"
                progress("prepare_training", fit_id=fit_id)
                train_x, train_y, weights, receipt = make_training_data(frame, labels, train, x, cfg)
                model = lgb.LGBMRegressor(**cfg["model_parameters"], random_state=seed)
                progress("training", fit_id=fit_id)
                began = time.monotonic()
                model.fit(train_x, train_y, sample_weight=weights)
                elapsed = time.monotonic() - began
                pred = np.clip(frame.baseline.to_numpy()[valid] + model.predict(x.loc[valid]), -5, 45)
                subset = eval_fold == spec["id"]
                out[subset] = pred
                supported_valid = valid & supported.to_numpy()
                stress_out[subset & stress_supported] = np.clip(altered.baseline.to_numpy()[supported_valid] + model.predict(stress_x.loc[supported_valid]), -5, 45)
                path = ARTIFACT / "04_models" / f"{fit_id}.txt"
                model.booster_.save_model(str(path))
                replay = lgb.Booster(model_file=str(path))
                replay_pred = np.clip(frame.baseline.to_numpy()[valid] + replay.predict(x.loc[valid], num_threads=2), -5, 45)
                discrepancy = float(np.max(np.abs(pred - replay_pred)))
                if discrepancy > 1e-12:
                    raise RuntimeError("tree replay mismatch")
                fits.append({"fit_id": fit_id, "fold": spec["id"], "seed": seed, "iterations": model.n_estimators_, "training": receipt, "runtime_seconds": elapsed, "replay_max_abs_error_C": discrepancy, "model_sha256": base.file_hash(path), "feature_count": len(x.columns)})
                base.atomic_json(ARTIFACT / "03_training/fit-receipts.json", fits)
                print(json.dumps({"completed": fit_id, "fits": len(fits), "runtime_seconds": elapsed}), flush=True)
            predictions[f"tree_seed{seed}"] = out
            predictions[f"fixed_half_seed{seed}"] = 0.5 * out + 0.5 * predictions[f"C_seed{seed}"]
            stress_predictions[f"tree_seed{seed}"] = stress_out
            stress_predictions[f"fixed_half_seed{seed}"] = 0.5 * stress_out + 0.5 * stress_predictions[f"C_seed{seed}"]
        first_seed = cfg["seeds"][0]
        run_seed(first_seed)
        screen = {name: a.metrics_by_scope(truth, predictions[f"{name}_seed{first_seed}"], eval_fold, primary) for name in cfg["policies"]}
        winner = min(screen, key=lambda name: (screen[name]["primary"]["rmse"], screen[name]["pooled"]["rmse"], cfg["policies"].index(name)))
        base.atomic_json(REPORT / "screen.json", {"winner": winner, "metrics": screen, "fits": len(fits)})
        if winner != "C":
            for seed in cfg["seeds"][1:]:
                run_seed(seed)
        confirmed_policies = ["C"] if winner == "C" else cfg["policies"]
        for name in confirmed_policies:
            predictions[f"{name}_mean"] = np.mean([predictions[f"{name}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
            stress_predictions[f"{name}_mean"] = np.mean([stress_predictions[f"{name}_seed{seed}"] for seed in cfg["seeds"]], axis=0)
        chosen, reference = predictions[f"{winner}_mean"], predictions["C_mean"]
        delta = base.metrics(truth[primary], chosen[primary])["rmse"] - base.metrics(truth[primary], reference[primary])["rmse"]
        masks = {"fall_all_supported": primary & stress_supported, "fall_masked_supported": primary & stress_supported & outage[evaluated]}
        metrics = {name: a.metrics_by_scope(truth, prediction, eval_fold, primary) for name, prediction in predictions.items()}
        stress_metrics = {scope: {name: base.metrics(truth[mask], pred[mask]) for name, pred in stress_predictions.items()} for scope, mask in masks.items()}
        output = ARTIFACT / "03_training/raw_oof.npz"
        np.savez_compressed(output, key=keys, truth=truth, fold=eval_fold, time=query.time.to_numpy(str), layer=query.layer.to_numpy(int), stress_supported=stress_supported, stress_selected=outage[evaluated], **predictions, **{f"stress_{name}": values for name, values in stress_predictions.items()})
        result = {"experiment_id": ID, "status": "PRIMARY_IMPROVEMENT_INTERNAL_ONLY" if delta < 0 else "NO_PRIMARY_IMPROVEMENT_P2_AB_COMPLETE", "winner_screen": winner, "chosen": f"{winner}_mean", "primary_delta_rmse_C": delta, "new_historical_fits": len(fits), "calibration_fits": 0, "fulltrain_fits": 0, "rows": len(truth), "training_eligible_rows": int(eligible.sum()), "feature_count": len(x.columns), "feature_names": list(x.columns), "metrics": metrics, "stress_metrics": stress_metrics, "runtime_seconds": time.monotonic() - started, "official_access_rows": 0, "csv_written": 0, "upload": 0, "official_control_rmse_C": 0.455143, "official_control_points": 27.622418, "new_expected_official_points": None, "raw_oof_sha256": base.file_hash(output), "sealed_hashes": seal["hashes"], "limitations": ["Repeated historical development; no fresh claim", "Intact primary autumn only one year", "Fixed half is untrained development diagnostic; no adaptive stacking", "Feature slots depend on nominal depth and preserve missingness", "Outage masking precedes every lag and baseline feature"]}
        base.atomic_json(REPORT / "result.json", result)
        base.atomic_json(ARTIFACT / "06_report/result.json", result)
        base.atomic_json(ARTIFACT / "progress.json", {"status": "COMPLETE", "fits": len(fits), "runtime_seconds": result["runtime_seconds"]})
        return result
    except Exception as error:
        receipt = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(error).__name__, "message": str(error), "fits": len(fits), "runtime_seconds": time.monotonic() - started, "automatic_restart": False}
        base.atomic_json(REPORT / "failure.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.seal and args.execute:
        raise ValueError("separate seal and execute")
    if args.seal:
        load_config()
        value = {"experiment_id": ID, "sealed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hashes": fingerprint(), "fit_cap": 9}
        REPORT.mkdir(parents=True, exist_ok=True)
        with SEAL.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        print(json.dumps(value, indent=2))
    elif args.execute:
        with threadpool_limits(limits=2):
            result = execute()
        print(json.dumps({key: result[key] for key in ("status", "chosen", "primary_delta_rmse_C", "new_historical_fits", "runtime_seconds")}))
    else:
        print(json.dumps({"config_valid": bool(load_config()), "execute": False}))


if __name__ == "__main__":
    main()

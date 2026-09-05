"""Exactly-once CPU wind-only dropout ablation; distributed train only.

No official-input, deployment, CSV, upload or full-fit branch exists. Historical
raw rows and models remain under ignored artifacts; reports contain aggregates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from run_p3_direct_sse_meta_20260905_v2 import bootstrap, slices, validate_frame  # noqa: E402
from run_p3_score_repair_20260905_v1 import aligned  # noqa: E402

from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds  # noqa: E402
from p3_wave.models import threshold_case_weights  # noqa: E402
from p3_wave.revin_patch import assign_storm_episodes_from_wave  # noqa: E402
from p3_wave.validation import expand_leads  # noqa: E402

NAME = "p3_wind_only_dropout_20260905_v4"
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
LOCK = ROOT / "artifacts" / f"{NAME}.ATTEMPT_LOCK.json"
KEYS = ["fold", "anchor_id", "station", "lead_h"]
LEADS = [3, 6, 9, 12, 18, 24]
WIND_PREFIXES = ("wspd_", "gust_", "wdir_sin_", "wdir_cos_", "wind_wave_alignment_", "wind_input_proxy_")
POLICIES = ["no_op", "control", "control_half", "wind_only", "wind_only_half"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value, *, progress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if progress else "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def stamp():
    return datetime.now(UTC).isoformat()


def official_zero():
    return dict.fromkeys(["test_context_rows", "test_index_rows", "sample_rows", "hidden_rows", "submission_csv_rows", "uploads", "external_rows"], 0)


def wind_columns(columns):
    return [column for column in columns if column.startswith(WIND_PREFIXES)]


def wind_observed(frame):
    columns = [column for column in wind_columns(frame.columns) if "_valid_" in column]
    if not columns:
        raise ValueError("wind support columns missing")
    return frame[columns].gt(0).any(axis=1).to_numpy()


def mask_wind(frame):
    masked = frame.copy()
    for column in wind_columns(frame.columns):
        masked[column] = 0.0 if "_valid_" in column else np.nan
    return masked


def augment_wind(frame, target, weights):
    target, weights = np.asarray(target), np.asarray(weights, float)
    if len(frame) != len(target) or len(frame) != len(weights):
        raise ValueError("augmentation shape mismatch")
    observed = wind_observed(frame)
    original = weights.copy()
    original[observed] *= 0.5
    matrix = pd.concat([frame, mask_wind(frame.loc[observed])], ignore_index=True)
    y = np.r_[target, target[observed]]
    weight = np.r_[original, weights[observed] * 0.5]
    indices = np.r_[np.arange(len(frame)), np.flatnonzero(observed)]
    discrepancy = float(np.max(np.abs(np.bincount(indices, weights=weight, minlength=len(frame)) - weights)))
    if discrepancy > 1e-12 or not np.array_equal(y, target[indices]):
        raise ValueError("original row weight or target changed")
    return matrix, y, weight, {"wind_observed_rows": int(observed.sum()), "original_rows": len(frame), "expanded_rows": len(matrix), "weight_sum_before": float(weights.sum()), "weight_sum_after": float(weight.sum()), "max_per_original_row_weight_error": discrepancy, "target_copy_exact": True, "nonwind_columns_preserved": True}


def categorical(matrix):
    matrix = matrix.copy()
    matrix["station"] = pd.Categorical(matrix.station, categories=["G-ORS", "I-ORS", "S-ORS"])
    matrix["lead_h"] = pd.Categorical(matrix.lead_h, categories=LEADS)
    if matrix[["station", "lead_h"]].isna().any().any():
        raise ValueError("unknown station or lead")
    return matrix


def install_guard(source, config):
    permitted = {(ROOT / name).resolve() for name in config["inputs"]}
    allowed_source = {source / name for name in config["source_files"]}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        owned = OUT in path.parents
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden input forbidden")
        if source in path.parents and (path not in allowed_source or (isinstance(args[1], str) and any(c in args[1] for c in "wax+"))):
            raise PermissionError("source not in immutable training allowlist")
        if path.suffix.lower() == ".csv" and path not in allowed_source:
            raise PermissionError("CSV forbidden")
        if path.suffix.lower() in {".parquet", ".npz", ".cbm", ".ckpt", ".pt", ".joblib", ".tabpfn_fit"} and path not in permitted and not owned:
            raise PermissionError("unapproved dataset/model forbidden")

    sys.addaudithook(guard)


def verify_inputs(config, source):
    verified = {}
    for name, expected in config["inputs"].items():
        verified[name] = sha(ROOT / name)
        if verified[name] != expected:
            raise ValueError(f"pinned dependency hash mismatch: {name}")
    for name, expected in config["source_files"].items():
        verified[f"source/{name}"] = sha(source / name)
        if verified[f"source/{name}"] != expected:
            raise ValueError("distributed train hash mismatch")
    return verified


def prepare(config, source):
    verified = verify_inputs(config, source)
    cache = ROOT / config["cache"]
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    keys = pd.read_parquet(cache / "validation_keys.parquet")
    oof = pd.read_parquet(cache / "oof.parquet")
    columns = json.loads((cache / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    wave = pd.read_csv(source / "train_wave.csv")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split = build_corrected_repeated_forward_folds(anchors, windows=config["windows"], gap_hours=78, footprint_hours=72)
    sort = ["anchor_id", "station", "fold"]
    actual, expected = selected.sort_values(sort), keys.sort_values(sort)
    if not actual[sort].reset_index(drop=True).equals(expected[sort].reset_index(drop=True)) or not np.array_equal(actual.episode_id, expected.episode_id):
        raise ValueError("recomputed validation population differs")
    if len(columns) != 591 or len(anchors) != 24360 or not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("cache schema mismatch")
    reference, chronology = validate_frame(oof, keys, anchors, config)
    lookup = features.set_index("anchor_id").loc[reference.anchor_id]
    reference["wind_observed"] = wind_observed(lookup)
    reference["onset_3h"] = lookup.hs_min_3h.lt(1.5).to_numpy()
    reference["nonwind_atmos_observed"] = lookup[[c for c in columns if c.startswith(("caph_", "airt_", "relh_")) and "_valid_" in c]].gt(0).any(axis=1).to_numpy()
    return features, anchors, columns, folds, reference, {"verified_inputs": verified, "split_audit": split, "reference_chronology": chronology, "wind_feature_columns": wind_columns(columns), "nonwind_features_preserved": [c for c in columns if c.startswith(("caph_", "airt_", "relh_"))], "onset_diagnostic": "current_hs>=1.5 and past3h observed hs_min<1.5; not exact first-crossing reconstruction; never training selection"}


def metrics(frame, prediction):
    result = slices(frame, np.asarray(prediction))
    truth = frame.target_hs.to_numpy()
    baseline = frame.final_prediction.to_numpy()
    for column in ("wind_observed", "onset_3h", "nonwind_atmos_observed"):
        result[f"by_{column}"] = {}
        for group, indices in frame.groupby(column).indices.items():
            sse = float(np.square(truth[indices] - prediction[indices]).sum())
            base_sse = float(np.square(truth[indices] - baseline[indices]).sum())
            result[f"by_{column}"][str(group)] = {"rows": len(indices), "cases": int(frame.iloc[indices].anchor_id.nunique()), "sse_m2": sse, "rmse_m": float(np.sqrt(sse / len(indices))), "baseline_rmse_m": float(np.sqrt(base_sse / len(indices))), "delta_rmse_m": float(np.sqrt(sse / len(indices)) - np.sqrt(base_sse / len(indices)))}
    errors = pd.DataFrame({"anchor_id": frame.anchor_id, "candidate_sse": np.square(truth - prediction), "baseline_sse": np.square(truth - baseline)}).groupby("anchor_id").sum()
    deltas = np.sqrt(errors.candidate_sse / 6) - np.sqrt(errors.baseline_sse / 6)
    result["case_delta_summary"] = {"improved_cases": int((deltas < 0).sum()), "worse_cases": int((deltas > 0).sum()), "unchanged_cases": int((deltas == 0).sum()), "median_delta_rmse_m": float(deltas.median()), "p90_delta_rmse_m": float(deltas.quantile(.9)), "worst_delta_rmse_m": float(deltas.max()), "used_for_automatic_rejection": False}
    return result


def execute():
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != NAME or config["arms"] != ["control", "wind_only"] or config["fit_budget"] != 6 or config["seed"] != 20260905:
        raise ValueError("contract changed")
    if LOCK.exists() or OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once output/lock exists; no restart")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    install_guard(source, config)
    features, anchors, columns, folds, reference, integrity = prepare(config, source)
    seal = {"created_utc": stamp(), "pid": os.getpid(), "runner_sha256": sha(Path(__file__)), "config_sha256": sha(CONFIG), "fit_budget": 6, "new_router_fits": 0, "official_access": official_zero(), "integrity": integrity}
    OUT.mkdir(parents=True)
    save(OUT / "seal.json", seal)
    save(LOCK, seal)
    save(REPORT / "preflight.json", seal)
    reference.to_parquet(OUT / "reference.parquet", index=False)
    fits, blocks = [], []
    process = psutil.Process()
    peak = 0.0

    def guard(_env=None):
        nonlocal peak
        peak = max(peak, process.memory_info().rss / 2**30)
        if time.perf_counter() - started > 1800 or peak > 8:
            raise RuntimeError("fixed 30-minute/8GiB budget exhausted; no restart")

    try:
        for fold in folds:
            train, y, train_meta = expand_leads(features, anchors, fold.train_ids, columns)
            valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, columns)
            valid = categorical(valid)
            valid_meta["fold"] = fold.name
            weights = threshold_case_weights(train_meta.current_hs.to_numpy())
            probe_path = OUT / f"{fold.name}_validation.parquet"
            valid.to_parquet(probe_path, index=False)
            for arm in config["arms"]:
                guard()
                if arm == "wind_only":
                    matrix, target, weight, augmentation = augment_wind(train, y, weights)
                else:
                    matrix, target, weight = train.copy(), y, weights
                    augmentation = {"original_rows": len(train), "expanded_rows": len(train), "weight_sum_before": float(weights.sum()), "weight_sum_after": float(weights.sum()), "max_per_original_row_weight_error": 0.0, "target_copy_exact": True}
                matrix = categorical(matrix)
                progress = {"status": "RUNNING", "pid": os.getpid(), "fold": fold.name, "arm": arm, "completed_fits": len(fits), "maximum_fits": 6, "elapsed_seconds": time.perf_counter() - started, "cpu_threads": 2, "gpu": False}
                save(OUT / "progress.json", progress, progress=True)
                print(json.dumps(progress), flush=True)
                begin = time.perf_counter()
                model = lgb.LGBMRegressor(**config["model"], random_state=config["seed"])
                model.fit(matrix, target, sample_weight=weight, categorical_feature=["station", "lead_h"], callbacks=[guard])
                residual = model.predict(valid)
                raw = valid_meta.current_hs.to_numpy() + residual
                if not np.isfinite(raw).all():
                    raise ValueError("nonfinite prediction")
                block = valid_meta.copy()
                block["arm"], block["prediction"], block["raw_prediction"] = arm, np.clip(raw, 0, 30), raw
                blocks.append(block)
                path = OUT / "models" / f"{fold.name}_{arm}.txt"
                path.parent.mkdir(exist_ok=True)
                model.booster_.save_model(str(path))
                restored = lgb.Booster(model_file=str(path))
                error = float(np.max(np.abs(restored.predict(valid, num_threads=2) - residual)))
                if error != 0:
                    raise ValueError("native save/reload differs")
                fits.append({"fold": fold.name, "arm": arm, "seed": config["seed"], "fit_number": len(fits) + 1, "fit_seconds": time.perf_counter() - begin, "train_cases": len(fold.train_ids), "validation_cases": len(fold.validation_ids), "augmentation": augmentation, "model_path": path.relative_to(ROOT).as_posix(), "model_sha256": sha(path), "probe_path": probe_path.relative_to(ROOT).as_posix(), "probe_sha256": sha(probe_path), "native_reload_max_abs_m": error})
                save(OUT / "fit-receipts.json", {"fits": fits}, progress=True)
                del model, restored, matrix
                gc.collect()
            del train, valid, train_meta
            gc.collect()
        oof = pd.concat(blocks, ignore_index=True)
        oof.to_parquet(OUT / "oof.parquet", index=False)
        policies = {"no_op": reference.final_prediction.to_numpy()}
        raw_metrics = {}
        for arm in config["arms"]:
            one = aligned(reference, oof.loc[oof.arm.eq(arm)], ["prediction", "raw_prediction"])
            policies[arm] = one.prediction.to_numpy()
            policies[f"{arm}_half"] = (policies["no_op"] + policies[arm]) / 2
            raw_metrics[arm] = metrics(reference, one.raw_prediction.to_numpy())
        all_metrics = {name: metrics(reference, policies[name]) for name in POLICIES}
        winner = min(POLICIES, key=lambda name: all_metrics[name]["rmse_m"])
        result = {"experiment_id": NAME, "status": "COMPLETE", "terminal": True, "completed_utc": stamp(), "pid": os.getpid(), "historical_fit_count": len(fits), "new_router_fit_count": 0, "runtime_seconds": time.perf_counter() - started, "peak_rss_gib": peak, "runner_sha256": sha(Path(__file__)), "config_sha256": sha(CONFIG), "seal_sha256": sha(OUT / "seal.json"), "oof_sha256": sha(OUT / "oof.parquet"), "reference_sha256": sha(OUT / "reference.parquet"), "fits": fits, "metrics": all_metrics, "raw_metrics": raw_metrics, "paired_wind_minus_control_delta_rmse_m": all_metrics["wind_only"]["rmse_m"] - all_metrics["control"]["rmse_m"], "selected_policy": winner, "selected_bootstrap": bootstrap(reference, policies[winner], config["bootstrap"]), "official_access": official_zero(), "submission_ready": False, "repeated_historical_surface_not_fresh_confirmation": True, "router_contract": "new_router_fits=0; exact clean baseline includes independently validated earlier-only router, no saved coefficient copying or new component substitution", "next_decision": "PRESERVE_NO_OP_NO_FULL_FIT" if winner == "no_op" else "INTERNAL_INFORMATION_CANDIDATE_REQUIRES_SEPARATE_FULLTRAIN_AND_APPROVAL"}
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", {"status": "COMPLETE", "result_sha256": sha(REPORT / "result.json"), "fit_count": len(fits), "runtime_seconds": result["runtime_seconds"]})
        print(json.dumps({"status": "COMPLETE", "fit_count": len(fits), "selected_policy": winner, "result_sha256": sha(REPORT / "result.json")}), flush=True)
    except Exception as exc:
        save(OUT / "terminal_result.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "message": str(exc), "completed_fits": len(fits), "elapsed_seconds": time.perf_counter() - started, "automatic_retry": False})
        raise


def replay():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    install_guard(source, config)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    if result["pid"] == os.getpid() or sha(Path(__file__)) != result["runner_sha256"] or sha(CONFIG) != result["config_sha256"]:
        raise ValueError("fresh-process frozen-source contract fails")
    verify_inputs(config, source)
    if sha(OUT / "oof.parquet") != result["oof_sha256"]:
        raise ValueError("saved OOF changed")
    oof = pd.read_parquet(OUT / "oof.parquet")
    checked = []
    for fit in result["fits"]:
        path, probe = ROOT / fit["model_path"], ROOT / fit["probe_path"]
        if sha(path) != fit["model_sha256"] or sha(probe) != fit["probe_sha256"]:
            raise ValueError("model or probe hash mismatch")
        valid = categorical(pd.read_parquet(probe))
        selected = oof[oof.fold.eq(fit["fold"]) & oof.arm.eq(fit["arm"])].reset_index(drop=True)
        if not np.array_equal(valid.lead_h.astype(int), selected.lead_h) or not np.array_equal(valid.station.astype(str), selected.station):
            raise ValueError("replay ordering mismatch")
        model = lgb.Booster(model_file=str(path))
        prediction = selected.current_hs.to_numpy() + model.predict(valid, num_threads=2)
        error = float(np.max(np.abs(prediction - selected.raw_prediction.to_numpy())))
        if error != 0:
            raise ValueError("fresh replay mismatch")
        checked.append({"fold": fit["fold"], "arm": fit["arm"], "rows": len(selected), "model_sha256": sha(path), "max_abs_raw_prediction_error_m": error})
    receipt = {"status": "PASS", "fresh_pid": os.getpid(), "training_pid": result["pid"], "models": len(checked), "rows_replayed": sum(x["rows"] for x in checked), "checks": checked, "result_sha256": sha(REPORT / "result.json"), "runner_sha256": sha(Path(__file__)), "official_access": official_zero(), "same_environment_fresh_process_only": True}
    save(REPORT / "fresh-process-replay.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    execute() if args.execute else replay()

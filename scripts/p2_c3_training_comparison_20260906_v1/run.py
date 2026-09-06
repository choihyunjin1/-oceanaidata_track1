"""New scratch CUDA C3 learning-amount/regularization study; training source only."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[2]
ID = "p2_c3_training_comparison_20260906_v1"
CONFIG = ROOT / "configs/experiments" / (ID + ".json")
REPORT = ROOT / "reports" / ID
OUT = ROOT / "artifacts" / ID
SEAL = REPORT / "preregistration-seal.json"
CORE = ROOT / "scripts/portable_20260906/P2/02_code/core.py"
spec = importlib.util.spec_from_file_location("p2_frozen_portable_core_comparison", CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

KEYS = ["station", "layer", "time"]
REQUIRED_COLUMNS = {"station", "year", "time", "layer", "depth", "nominal_depth", "temp", "psal"}
RECIPE_IDS = ("C60", "L120", "D60")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, progress=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if progress else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def cfg():
    value = read(CONFIG)
    if value["experiment_id"] != ID or value["maximum_fits"] != 28 or value["max_seconds"] != 5400:
        raise ValueError("frozen study resource contract changed")
    if value["recipes"] != [{"id": "C60", "epochs": 60, "weight_decay": 0.0001},
                            {"id": "L120", "epochs": 120, "weight_decay": 0.0001},
                            {"id": "D60", "epochs": 60, "weight_decay": 0.001}]:
        raise ValueError("fixed learning-amount/regularization recipes changed")
    if value["seeds"] != [20260901, 20260902, 20260903] or sha(CORE) != value["core_sha256"]:
        raise ValueError("C3 seed/core lineage changed")
    if value["base_recipe"]["cpu_threads"] != 2 or value["base_recipe"]["device"] != "cuda":
        raise ValueError("same-resource CUDA CPU2 contract required")
    if any(value[k] != 0 for k in ("official_rows", "hidden_rows", "csv_written", "upload")):
        raise ValueError("training-only scope required")
    return value


def contract():
    value = read(ROOT / cfg()["evaluation_contract"])["P2"]
    if len(value["folds"]) != 8 or value["purge_days"] != 7 or value["primary"]["fold"] != "B3":
        raise ValueError("exact v5 eight-fold/7day/B3 contract required")
    return value


def pins():
    paths = [Path(__file__), CORE, CONFIG, ROOT / "configs/evaluation/ocean_forward_v5.json",
             Path(__file__).with_name("qa.py"),
             ROOT / "tests/test_p2_c3_training_comparison_20260906_v1.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def seal():
    cfg()
    contract()
    save(SEAL, {"status": "PREREGISTERED_BEFORE_NEW_FITS", "created_unix": time.time(),
        "pins": pins(), "packages": {p: importlib.metadata.version(p) for p in
        ("numpy", "pandas", "torch", "pyarrow", "scikit-learn", "threadpoolctl")},
        "GPU_authorized_by_seal": False, "new_fits": 0})


def verify_seal():
    frozen = read(SEAL)
    if frozen["pins"] != pins() or any(importlib.metadata.version(k) != v for k, v in frozen["packages"].items()):
        raise ValueError("code/config/environment seal drift")
    return cfg()


def key_values(frame):
    if frame[KEYS].isna().any().any():
        raise ValueError("missing keys")
    result = (frame.station.astype(str) + "|" + frame.layer.astype(int).astype(str) + "|" +
              pd.to_datetime(frame.time, utc=True).astype(str)).to_numpy(str)
    if len(np.unique(result)) != len(result):
        raise ValueError("duplicate keys")
    return result


def array_sha(values):
    value = np.ascontiguousarray(values)
    digest = hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def feature_hash(frame):
    return [array_sha(v) for v in core.arrays(frame)]


def source_path():
    return (Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv")


def path_allowed(path, writing, source, output):
    path = Path(path).resolve()
    if "external_data" in path.parts or "hidden" in path.name.lower():
        return False
    if path == source:
        return not writing
    if path.suffix.lower() == ".csv":
        return False
    if path.suffix.lower() in {".pt", ".pth", ".npz", ".npy", ".parquet", ".pkl", ".pickle", ".cbm"}:
        return output in path.parents
    return True


def install_guard(source, output, training):
    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network forbidden")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        writing = (isinstance(args[1], str) and any(c in args[1] for c in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR)))
        if not path_allowed(args[0], writing, source, output):
            raise PermissionError("source-only/own-artifact path guard")
    sys.addaudithook(guard)
    if training:
        def denied(*_args, **_kwargs):
            raise PermissionError("torch.load forbidden during fresh training")
        torch.load = denied


def prepare_frame(observations):
    obs = observations.copy()
    target = obs.layer.isin(core.TARGET_LAYERS)
    truth = obs.loc[target, "temp"].to_numpy(float).copy()
    obs.loc[target, ["temp", "psal"]] = np.nan
    frame, placeholder = core.public_frame(obs)
    if not np.isnan(placeholder).all():
        raise ValueError("target temperature/salinity firewall failed")
    keep = np.isfinite(truth) & np.isfinite(frame.baseline) & frame.public_temp_count.ge(2).to_numpy()
    return frame.loc[keep].reset_index(drop=True), truth[keep]


def load_source(config):
    path = source_path()
    if sha(path) != config["source_sha256"]:
        raise ValueError("distributed source SHA changed")
    obs = pd.read_csv(path)
    if set(obs.columns) != REQUIRED_COLUMNS or len(obs) != config["expected_source_rows"] or obs.duplicated(KEYS).any():
        raise ValueError("source schema/population/keys changed")
    if set(obs.station) != {"S-ORS"}:
        raise ValueError("source station changed")
    obs.time = pd.to_datetime(obs.time, utc=True)
    obs = obs.loc[obs.time.ge(pd.Timestamp(config["train_start"])) & obs.time.lt(pd.Timestamp(config["train_stop"]))].copy()
    frame, truth = prepare_frame(obs)
    if len(frame) != config["expected_eligible_rows"]:
        raise ValueError("eligible target population changed")
    key_values(frame)
    return frame, truth


def split_masks(frame, fold, purge_days=7):
    times = pd.to_datetime(frame.time, utc=True)
    left, right = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    purge = pd.Timedelta(days=purge_days)
    training = ((times < left - purge) | (times >= right + purge)).to_numpy()
    valid = ((times >= left) & (times < right)).to_numpy()
    if not training.any() or not valid.any() or (training & valid).any():
        raise ValueError("unsupported or overlapping fixed split")
    return training, valid


def make_outage(frame, fold):
    start, stop = pd.Timestamp(fold["end"]) - pd.Timedelta(days=17), pd.Timestamp(fold["end"])
    stamp = pd.to_datetime(frame.time, utc=True)
    selected = (stamp.ge(start) & stamp.lt(stop)).to_numpy()
    altered = frame.copy()
    altered.loc[selected, ["temp_5", "psal_5"]] = np.nan
    altered = core.refresh_public(altered)
    supported = np.isfinite(altered.baseline) & altered.public_temp_count.ge(2).to_numpy()
    return altered, selected, np.asarray(supported, dtype=bool)


def metric(truth, pred):
    y, p = np.asarray(truth), np.asarray(pred)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("same-row finite metric contract required")
    n = len(y)
    sse = float(np.dot(p - y, p - y))
    return {"n": n, "sse": sse, "rmse_C": float(np.sqrt(sse / n)) if n else None,
            "bias_C": float(np.mean(p - y)) if n else None}


def predict(model, frame):
    normalized = core.predict_model(model, *core.arrays(frame))
    return frame.baseline.to_numpy(float) + core.compute_profile_scale(frame) * normalized


def recipe_config(config, recipe):
    return config["base_recipe"] | {"epochs": recipe["epochs"], "weight_decay": recipe["weight_decay"]}


def support_payload(frame, truth):
    details, selected = [], np.zeros(len(frame), dtype=np.int8)
    for fold in contract()["folds"]:
        train, valid = split_masks(frame, fold)
        selected += valid.astype(np.int8)
        query = frame.loc[valid].reset_index(drop=True)
        outage, affected, supported = make_outage(query, fold)
        details.append({"fold": fold["id"], "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
            "train_keys_sha256": array_sha(key_values(frame.loc[train])),
            "valid_keys_sha256": array_sha(key_values(query)), "truth_sha256": array_sha(truth[valid]),
            "feature_hash": feature_hash(query), "outage_feature_hash": feature_hash(outage) if supported.all() else None,
            "outage_rows": int(affected.sum()), "unsupported_outage_rows": int((~supported).sum()),
            "natural_T5_missing_rows": int((~np.isfinite(query.temp_5)).sum()),
            "target_actual_depth_missing_rows": int((~np.isfinite(query.target_actual_depth)).sum()),
            "outage_status": "SUPPORT_BLOCKED" if not supported.all() else (
                "SUPPORTED" if affected.any() else "NOT_ESTIMABLE_NO_ROWS"),
            "validation_rows_deleted": 0})
    if not np.all(selected == 1):
        raise ValueError("eight-fold OOF ownership must cover every eligible row exactly once")
    return {"eligible_rows": len(frame), "folds": details, "target_mask": "all2/3/4 temp+psal before features",
            "source_sha256": sha(source_path()), "new_fits": 0, "official_rows": 0}


def prepare():
    config = verify_seal()
    install_guard(source_path(), OUT, False)
    frame, truth = load_source(config)
    save(REPORT / "source-support.json", support_payload(frame, truth))
    print(json.dumps({"status": "SOURCE_SUPPORT_READY", "rows": len(frame), "fits": 0}))


def pilot_estimate(seconds, config):
    # Full screen is 32 sixty-epoch equivalents, confirmation at most another 6.
    return seconds * 38 * config["pilot"]["safety_factor"] + config["pilot"]["remaining_nonfit_allowance_seconds"]


def choose_challenger(screen):
    return min(RECIPE_IDS[1:], key=lambda arm: (screen["natural"][arm]["B3_primary"]["rmse_C"],
               screen["natural"][arm]["all8_pooled"]["rmse_C"], RECIPE_IDS.index(arm)))


def build_summary(truth, natural, outage, support, keyfold, layer, time_values):
    output = {"natural": {}, "outage": {}, "slices": {}}
    for arm in natural:
        output["natural"][arm] = {"all8_pooled": metric(truth, natural[arm]),
                                   "B3_primary": metric(truth[keyfold == "B3"], natural[arm][keyfold == "B3"])}
        output["outage"][arm] = {}
        output["slices"][arm] = []
        for spec in support["folds"]:
            selected = keyfold == spec["fold"]
            output["natural"][arm][spec["fold"]] = metric(truth[selected], natural[arm][selected])
            if spec["outage_status"] == "SUPPORT_BLOCKED":
                output["outage"][arm][spec["fold"]] = {"status": "SUPPORT_BLOCKED", "rows_deleted": 0,
                    "unsupported_rows": spec["unsupported_outage_rows"], "metrics": None}
            else:
                times = pd.to_datetime(time_values, utc=True)
                fold = next(f for f in contract()["folds"] if f["id"] == spec["fold"])
                interval = selected & (times >= pd.Timestamp(fold["end"]) - pd.Timedelta(days=17)) & (times < pd.Timestamp(fold["end"]))
                output["outage"][arm][spec["fold"]] = {"status": spec["outage_status"], "rows_deleted": 0,
                    "whole_fold": metric(truth[selected], outage[arm][selected]),
                    "interval": metric(truth[interval], outage[arm][interval])}
            for target_layer in (2, 3, 4):
                rows = selected & (layer == target_layer)
                output["slices"][arm].append({"fold": spec["fold"], "layer": target_layer,
                    **metric(truth[rows], natural[arm][rows])})
    return output


def train_worker():
    config = verify_seal()
    receipt = read(OUT / "ATTEMPT_LOCK.json")
    if not receipt.get("gpu_exclusive_handoff_confirmed") or receipt["support_sha256"] != sha(REPORT / "source-support.json"):
        raise ValueError("GPU handoff/source-support lock missing or changed")
    receipt.update(worker_pid=os.getpid(), completed_fits=0, attempted_fits=0, fits=[],
                   official_rows=0, hidden_rows=0, csv_written=0, upload=0)
    started = receipt["started_unix"]

    def expired():
        save(OUT / "terminal_result.json", {"status": "BUDGET_WALL_CAP_TERMINAL", "completed_fits": receipt["completed_fits"],
            "attempted_fits": receipt["attempted_fits"], "automatic_restart": False})
        os._exit(124)
    timer = threading.Timer(max(0, config["max_seconds"] - (time.time() - started)), expired)
    timer.daemon = True
    timer.start()

    def progress(stage, epoch=None, fit_seconds=None):
        save(OUT / "progress.json", {"stage": stage, "pid": os.getpid(), "completed_fits": receipt["completed_fits"],
            "maximum_fits": 28, "epoch": epoch, "fit_seconds": fit_seconds, "runtime_seconds": time.time() - started}, progress=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
        install_guard(source_path(), OUT, True)
        frame, truth = load_source(config)
        support = support_payload(frame, truth)
        if support != read(REPORT / "source-support.json"):
            raise ValueError("sealed prepare support changed")
        (OUT / "03_model").mkdir()
        (OUT / "04_evaluation").mkdir()
        natural = {arm: np.full(len(frame), np.nan) for arm in RECIPE_IDS}
        outage_pred = {arm: np.full(len(frame), np.nan) for arm in RECIPE_IDS}
        folds = np.full(len(frame), "", dtype="U2")

        def fit_one(fold, recipe, seed, training, labels, query, altered, supported):
            name = f"{recipe['id']}_{fold['id']}_seed{seed}"
            recipe_cfg = recipe_config(config, recipe)
            data, mass = core.training_arrays(training, labels, "v23_blockmask", recipe_cfg)
            receipt["attempted_fits"] += 1
            if receipt["attempted_fits"] > config["maximum_fits"]:
                raise ValueError("fit budget exhausted")
            model, fit = core.fit_model(data, "v23_blockmask", seed, recipe_cfg,
                lambda epoch, elapsed: progress(name, epoch, elapsed))
            model_path = OUT / "03_model" / (name + ".pt")
            with model_path.open("xb") as handle:
                torch.save(model.state_dict(), handle)
            fit.update(fit_id=name, recipe=recipe['id'], fold=fold['id'], model_file=model_path.relative_to(OUT).as_posix(),
                model_sha256=sha(model_path), training=mass, train_keys_sha256=array_sha(key_values(training)),
                train_truth_sha256=array_sha(labels), valid_keys_sha256=array_sha(key_values(query)),
                training_arrays_sha256=[array_sha(v) for v in data], recipe_config=recipe_cfg)
            if receipt["completed_fits"] == 0:
                estimate = pilot_estimate(fit["runtime_seconds"], config)
                save(OUT / "pilot-resource.json", {"fit_id": name, "fit_seconds": fit["runtime_seconds"],
                    "predicted_total_seconds": estimate, "cap_seconds": config["max_seconds"], "first_fit_reused": True})
                if estimate > config["max_seconds"]:
                    receipt["completed_fits"] += 1
                    receipt["fits"].append(fit)
                    raise RuntimeError("PILOT_PROJECTED_BUDGET_EXCEEDED_BEFORE_PERFORMANCE")
            values = predict(model, query)
            stress = predict(model, altered) if supported.all() else None
            payload = {"natural": values}
            if stress is not None:
                payload["outage"] = stress
            path = OUT / "04_evaluation" / (name + ".npz")
            np.savez(path, **payload)
            fit["predictions_file"], fit["predictions_sha256"] = path.relative_to(OUT).as_posix(), sha(path)
            receipt["fits"].append(fit)
            receipt["completed_fits"] += 1
            save(OUT / "fit-receipts.json", receipt["fits"], progress=True)
            progress(name + "_COMPLETE")
            print(json.dumps({"fit_complete": name, "completed_fits": receipt["completed_fits"]}), flush=True)
            del model, data
            torch.cuda.empty_cache()
            return values, stress

        for fold in contract()["folds"]:
            train, valid = split_masks(frame, fold)
            training, query = frame.loc[train].reset_index(drop=True), frame.loc[valid].reset_index(drop=True)
            altered, _, supported = make_outage(query, fold)
            folds[valid] = fold["id"]
            for recipe in config["recipes"]:
                values, stress = fit_one(fold, recipe, config["seeds"][0], training, truth[train], query, altered, supported)
                natural[recipe["id"]][valid] = values
                if stress is not None:
                    outage_pred[recipe["id"]][valid] = stress
        if receipt["completed_fits"] != 24:
            raise ValueError("screen fit count mismatch")
        screen = build_summary(truth, natural, outage_pred, support, folds, frame.layer.to_numpy(), frame.time.to_numpy())
        winner = choose_challenger(screen)
        save(OUT / "screen-selection.json", {"selected_challenger": winner, "status": "FIRST_SEED_RETROSPECTIVE_SELECTION", "screen": screen})
        primary = next(f for f in contract()["folds"] if f["id"] == "B3")
        train, valid = split_masks(frame, primary)
        training, query = frame.loc[train].reset_index(drop=True), frame.loc[valid].reset_index(drop=True)
        altered, _, supported = make_outage(query, primary)
        confirmation = {}
        for arm in ("C60", winner):
            recipe = next(r for r in config["recipes"] if r["id"] == arm)
            predictions = [natural[arm][valid]]
            stress_predictions = [outage_pred[arm][valid]]
            for seed in config["seeds"][1:]:
                values, stress = fit_one(primary, recipe, seed, training, truth[train], query, altered, supported)
                predictions.append(values)
                stress_predictions.append(stress if stress is not None else np.full(len(query), np.nan))
            confirmation[arm] = {"natural_all3": np.mean(predictions, axis=0),
                "natural_heldback2": np.mean(predictions[1:], axis=0),
                "outage_all3": np.mean(stress_predictions, axis=0),
                "outage_heldback2": np.mean(stress_predictions[1:], axis=0)}
        arrays = {"key": key_values(frame), "truth": truth, "fold": folds, "layer": frame.layer.to_numpy(),
                  "time": frame.time.to_numpy(str), "natural_T5_missing": ~np.isfinite(frame.temp_5.to_numpy()),
                  **{f"natural_{k}": v for k, v in natural.items()},
                  **{f"outage_{k}": v for k, v in outage_pred.items()}}
        arrays.update({f"B3_{arm}_{name}": pred for arm, values in confirmation.items() for name, pred in values.items()})
        np.savez(OUT / "evaluation.npz", **arrays)
        confirmation_metrics = {arm: {name: metric(truth[valid], pred) if np.isfinite(pred).all() else None
                                for name, pred in values.items()} for arm, values in confirmation.items()}
        delta = confirmation_metrics[winner]["natural_all3"]["rmse_C"] - confirmation_metrics["C60"]["natural_all3"]["rmse_C"]
        verify_seal()
        if sha(source_path()) != config["source_sha256"]:
            raise ValueError("source changed during training")
        receipt.update(status="COMPLETE_24_SCREEN_PLUS_4_PRIMARY_SEED_FITS", support=support,
            screen=screen, selected_challenger=winner, confirmation=confirmation_metrics,
            primary_three_seed_delta_rmse_C=delta, retained=delta < 0,
            retention_status="PRIMARY_MEAN_IMPROVED_CANDIDATE" if delta < 0 else "NO_PRIMARY_THREE_SEED_MEAN_IMPROVEMENT",
            confirmation_is_new_holdout=False, all8fold3seed_evaluated=False,
            source_sha256=sha(source_path()), evaluation_sha256=sha(OUT / "evaluation.npz"),
            runtime_seconds=time.time() - started, device="cuda", cpu_threads=2,
            gpu_name=torch.cuda.get_device_name(0), source_module_sha256=sha(CORE))
        if receipt["completed_fits"] != receipt["attempted_fits"] or receipt["completed_fits"] != 28:
            raise ValueError("28 actual fit accounting mismatch")
        save(OUT / "terminal_result.json", receipt)
        print(json.dumps({"status": receipt["status"], "completed_fits": 28, "runtime_seconds": receipt["runtime_seconds"]}), flush=True)
    except BaseException as error:
        if not (OUT / "terminal_result.json").exists():
            receipt.update(status="BUDGET_PROJECTION_TERMINAL" if "PILOT_PROJECTED" in str(error) else "TERMINAL_TECHNICAL_FAILURE",
                error_type=type(error).__name__, error=str(error), automatic_restart=False, runtime_seconds=time.time() - started)
            save(OUT / "terminal_result.json", receipt)
        raise
    finally:
        timer.cancel()


def replay():
    started = time.monotonic()
    config = verify_seal()
    result = read(OUT / "terminal_result.json")
    if result["status"] != "COMPLETE_24_SCREEN_PLUS_4_PRIMARY_SEED_FITS" or result["completed_fits"] != 28:
        raise ValueError("complete training required")
    if os.getpid() == result["worker_pid"] or (OUT / "fresh-replay.json").exists():
        raise ValueError("fresh independent PID and new replay receipt required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA replay required; CPU fallback forbidden")
    save(OUT / "REPLAY_ATTEMPT_LOCK.json", {"pid": os.getpid(), "training_pid": result["worker_pid"],
        "terminal_result_sha256": sha(OUT / "terminal_result.json"), "started_unix": time.time()})
    install_guard(source_path(), OUT, False)
    frame, truth = load_source(config)
    checks = []
    for fold in contract()["folds"]:
        training, valid = split_masks(frame, fold)
        query = frame.loc[valid].reset_index(drop=True)
        altered, _, supported = make_outage(query, fold)
        for fit in [r for r in result["fits"] if r["fold"] == fold["id"]]:
            path = OUT / fit["model_file"]
            if sha(path) != fit["model_sha256"] or sha(OUT / fit["predictions_file"]) != fit["predictions_sha256"]:
                raise ValueError("own model/prediction hash mismatch")
            if array_sha(key_values(frame.loc[training])) != fit["train_keys_sha256"] or array_sha(truth[training]) != fit["train_truth_sha256"]:
                raise ValueError("training lineage changed")
            model = core.make_model("v23_blockmask", 11)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            stored = np.load(OUT / fit["predictions_file"], allow_pickle=False)
            if not np.array_equal(predict(model, query), stored["natural"]):
                raise ValueError("whole-validation natural replay mismatch")
            if supported.all() and not np.array_equal(predict(model, altered), stored["outage"]):
                raise ValueError("whole-validation outage replay mismatch")
            checks.append(fit["fit_id"])
            stored.close()
            del model
            torch.cuda.empty_cache()
    verify_seal()
    if sha(source_path()) != config["source_sha256"] or sha(OUT / "evaluation.npz") != result["evaluation_sha256"]:
        raise ValueError("source/evaluation drift during replay")
    save(OUT / "fresh-replay.json", {"status": "PASS", "pid": os.getpid(), "training_pid": result["worker_pid"],
        "full_models_replayed": len(checks), "checks": checks, "replay_scope": "all validation rows for each of28 fits; natural+supported_outage",
        "maximum_absolute_error": 0, "fits": 0, "official_rows": 0, "device": "cuda", "cpu_threads": 2,
        "runtime_seconds": time.monotonic() - started, "terminal_result_sha256": sha(OUT / "terminal_result.json")})
    print(json.dumps({"status": "PASS", "full_models_replayed": len(checks), "fits": 0}))


def launch(gpu_authorized):
    config = verify_seal()
    if not gpu_authorized:
        raise PermissionError("root exclusive GPU handoff required before launch")
    if OUT.exists() or not (REPORT / "source-support.json").exists():
        raise ValueError("new output and prior source-support QA required")
    OUT.mkdir(parents=True)
    save(OUT / "ATTEMPT_LOCK.json", {"id": ID, "started_unix": time.time(), "launcher_pid": os.getpid(),
        "seal_sha256": sha(SEAL), "support_sha256": sha(REPORT / "source-support.json"),
        "maximum_fits": 28, "gpu_exclusive_handoff_confirmed": True})
    child = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "worker"])
    try:
        code = child.wait(timeout=config["max_seconds"])
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
        child.wait(timeout=30)
        raise SystemExit(124) from None
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "prepare", "execute", "worker", "replay"))
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.mode == "seal":
            seal()
        elif args.mode == "prepare":
            prepare()
        elif args.mode == "execute":
            launch(args.gpu_authorized)
        elif args.mode == "worker":
            train_worker()
        else:
            replay()

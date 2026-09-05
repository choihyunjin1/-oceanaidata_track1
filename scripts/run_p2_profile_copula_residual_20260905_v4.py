"""Isolated train-only physical-profile copula correction of eligible C3."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_objective_alignment_20260905_v2 as previous  # noqa: E402

base = previous.base
ID = "p2_profile_copula_residual_20260905_v4"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
SEAL = REPORT / "preregistration-seal.json"
OLD_RAW = previous.ARTIFACT / "03_training/raw_oof.npz"


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID
    assert cfg["seeds"] == [20260901, 20260902, 20260903]
    assert cfg["policies"] == {"C": 0.0, "full": 1.0, "half": 0.5}
    assert cfg["deterministic_unique_residual_fits"] == 3
    assert cfg["quadrature_nodes"] == 15 and cfg["feature_dependency_hours"] == 0
    assert cfg["cpu_threads"] == 2 and not cfg["gpu_used"]
    return cfg


def manifests():
    receipts = json.loads((base.REPORT / "result.json").read_text(encoding="utf-8"))["fit_receipts"]
    rows = []
    for spec in previous.load_config()["folds"]:
        for seed in read_config()["seeds"]:
            fit_id = f"v23_blockmask_{seed}_{spec['id']}"
            receipt = next(r for r in receipts if r["fit_id"] == fit_id)
            path = previous.OLD / f"{fit_id}.pt"
            assert base.file_hash(path) == receipt["model_sha256"]
            rows.append({"fold": spec["id"], "seed": seed, "path": str(path.relative_to(ROOT)), "sha256": receipt["model_sha256"]})
    return rows


def fingerprint():
    cfg = read_config()
    files = ["scripts/run_p2_objective_alignment_20260905_v2.py", "configs/experiments/p2_objective_alignment_20260905_v2.json", *previous.DEPENDENCIES, cfg["episodes_source"]]
    assert base.file_hash(OLD_RAW) == cfg["control_raw_sha256"]
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG), "dependencies": {p: base.file_hash(ROOT / p) for p in files}, "control_raw": cfg["control_raw_sha256"], "models": manifests()}


def actual_profile(frame):
    """Actual-depth piecewise public temperature profile; duplicate depths averaged."""
    nominal = frame.target_depth.to_numpy(float)
    actual = frame.target_actual_depth.to_numpy(float)
    target = np.where(np.isfinite(actual) & (actual > 0), actual, nominal)
    temperatures = frame[[f"temp_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    actuals = frame[[f"depth_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    nominals = frame[[f"nominal_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    depths = np.where(np.isfinite(actuals) & (actuals > 0), actuals, nominals)
    interpolated, gradient = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for index in range(len(frame)):
        good = np.isfinite(temperatures[index]) & np.isfinite(depths[index])
        z, inv = np.unique(depths[index, good], return_inverse=True)
        if not len(z):
            continue
        temp = np.bincount(inv, weights=temperatures[index, good]) / np.bincount(inv)
        interpolated[index] = np.interp(target[index], z, temp)
        gradient[index] = 0.0
        if len(z) >= 2 and z[0] <= target[index] < z[-1]:
            left = int(np.searchsorted(z, target[index], side="right") - 1)
            gradient[index] = (temp[left + 1] - temp[left]) / (z[left + 1] - z[left])
    return target, interpolated, gradient


def physical_features(frame, c):
    if len(frame) != len(c) or not np.isfinite(c).all():
        raise ValueError("C3 requires aligned finite predictions")
    target, interp, gradient = actual_profile(frame)
    nominal = frame.target_depth.to_numpy(float)
    columns = [c, target, nominal, interp - frame.baseline.to_numpy(float), gradient]
    for variable, left, right in (("temp", 1, 5), ("temp", 1, 6), ("temp", 5, 6), ("psal", 1, 5), ("psal", 5, 6)):
        columns.append(frame[f"{variable}_{left}"].to_numpy(float) - frame[f"{variable}_{right}"].to_numpy(float))
    columns.append(target - nominal)
    result = np.column_stack(columns)
    if np.isinf(result).any():
        raise ValueError("infinite physical input")
    return result


def latent(values, sorted_values):
    result = np.zeros(len(values))
    present = np.isfinite(values)
    if not len(sorted_values):
        return result
    left = np.searchsorted(sorted_values, values[present], side="left")
    right = np.searchsorted(sorted_values, values[present], side="right")
    p = (left + right) / (2 * len(sorted_values))
    p = np.clip(p, 0.5 / len(sorted_values), 1 - 0.5 / len(sorted_values))
    result[present] = ndtri(p)
    return result


def fit_copula(x, residual):
    if len(x) != len(residual) or not np.isfinite(residual).all():
        raise ValueError("finite aligned training residual required")
    marginals = [np.sort(col[np.isfinite(col)]) for col in x.T]
    response = np.sort(residual)
    z = np.column_stack([latent(x[:, i], marginal) for i, marginal in enumerate(marginals)] + [latent(residual, response)])
    fitted = LedoitWolf().fit(z)
    return {"location": fitted.location_, "covariance": fitted.covariance_, "shrinkage": np.array(fitted.shrinkage_), "response": response, **{f"marginal_{i}": v for i, v in enumerate(marginals)}}


def predict_copula(model, x):
    d = x.shape[1]
    present = np.isfinite(x)
    for i in range(d):
        present[:, i] &= len(model[f"marginal_{i}"]) > 0
    z = np.column_stack([latent(x[:, i], model[f"marginal_{i}"]) for i in range(d)])
    locations, cov = model["location"], model["covariance"]
    nodes, weights = np.polynomial.hermite.hermgauss(15)
    support = (np.arange(len(model["response"])) + 0.5) / len(model["response"])
    result = np.empty(len(x))
    for pattern in np.unique(present, axis=0):
        rows = np.all(present == pattern, axis=1)
        observed = np.flatnonzero(pattern)
        if len(observed):
            beta = np.linalg.pinv(cov[np.ix_(observed, observed)], hermitian=True) @ cov[observed, d]
            mean = locations[d] + (z[rows][:, observed] - locations[observed]) @ beta
            variance = max(0.0, float(cov[d, d] - cov[d, observed] @ beta))
        else:
            mean, variance = np.full(int(rows.sum()), locations[d]), max(0.0, float(cov[d, d]))
        probability = ndtr(mean[:, None] + np.sqrt(2 * variance) * nodes[None, :])
        transformed = np.interp(probability, support, model["response"])
        result[rows] = transformed @ (weights / np.sqrt(np.pi))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite correction")
    return result


def episode_frame(frame, episode):
    times = pd.to_datetime(frame.time, utc=True)
    selected = np.asarray((times >= base.utc(episode["start"])) & (times < base.utc(episode["stop"])))
    changed = frame.copy()
    changed.loc[selected, ["temp_5", "psal_5"]] = np.nan
    return base.refresh_public(changed), selected


def metric_table(truth, predictions, scopes):
    result = {}
    for name, mask in scopes.items():
        values = {arm: base.metrics(truth[mask], pred[mask]) for arm, pred in predictions.items()} if mask.any() else None
        result[name] = {"n": int(mask.sum()), "metrics": values}
    return result


def install_guard(sealed):
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    approved = {(ROOT / row["path"]).resolve() for row in sealed["models"]}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network prohibited")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("forbidden external or hidden path")
        if path.suffix.lower() == ".csv" and path != source:
            raise PermissionError("only distributed observations CSV allowed")
        if path == source and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
            raise PermissionError("immutable source")
        if path.suffix.lower() == ".pt" and path not in approved:
            raise PermissionError("unapproved model")
        if path.suffix.lower() == ".npz" and path != OLD_RAW.resolve() and OUT not in path.parents:
            raise PermissionError("unapproved arrays")
    sys.addaudithook(guard)


def policies(c, residual, cfg):
    return {name: c.copy() if weight == 0 else c + weight * residual for name, weight in cfg["policies"].items()}


def execute():
    cfg = read_config()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    if sealed["hashes"] != fingerprint():
        raise ValueError("seal mismatch")
    if OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once attempt exists")
    install_guard(sealed["hashes"])
    OUT.mkdir(parents=True)
    save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "sealed": sealed, "new_historical_cap": 6})
    started = time.monotonic()
    fits = []
    try:
        torch.set_num_threads(2)
        frame, labels = previous.load_data(previous.load_config())
        assignment = np.full(len(frame), "", dtype="U20")
        for spec in previous.load_config()["folds"]:
            _, validation = base.restoration_masks(frame.time, spec, 7)
            assignment[validation] = spec["id"]
        used = assignment != ""
        query, truth, fold = frame.loc[used].reset_index(drop=True), labels[used], assignment[used]
        key = (query.time.astype(str) + "|" + query.layer.astype(str)).to_numpy(str)
        old = np.load(OLD_RAW, allow_pickle=False)
        for name, value in (("key", key), ("truth", truth), ("fold", fold)):
            assert np.array_equal(value, old[name])
        assert len(truth) == 69850 and int((fold == cfg["primary_fold"]).sum()) == 26273
        models = {}
        for row in sealed["hashes"]["models"]:
            model = base.make_model("v23", 11)
            model.load_state_dict(torch.load(ROOT / row["path"], map_location="cpu", weights_only=True))
            models[row["fold"], row["seed"]] = model.eval()

        def cmean(local, fold_name):
            return np.mean(np.stack([previous.predict_absolute(models[fold_name, seed], local) for seed in cfg["seeds"]]), axis=0)

        c, correction = np.full(len(truth), np.nan), np.full(len(truth), np.nan)
        learned, replay = {}, {}
        for spec in previous.load_config()["folds"]:
            train, validation = base.restoration_masks(frame.time, spec, 7)
            assert not np.any(train & validation)
            train_frame = frame.loc[train].reset_index(drop=True)
            train_c = cmean(train_frame, spec["id"])
            x_train = physical_features(train_frame, train_c)
            fit_start = time.monotonic()
            model = fit_copula(x_train, labels[train] - train_c)
            fit_time = time.monotonic() - fit_start
            model_file = OUT / f"copula_{spec['id']}.npz"
            np.savez_compressed(model_file, **model)
            loaded = dict(np.load(model_file, allow_pickle=False))
            rows = fold == spec["id"]
            local = query.loc[rows].reset_index(drop=True)
            c[rows] = cmean(local, spec["id"])
            for seed in cfg["seeds"]:
                assert np.array_equal(previous.predict_absolute(models[spec["id"], seed], local), old[f"C_seed{seed}"][rows])
            x = physical_features(local, c[rows])
            correction[rows] = predict_copula(model, x)
            assert np.array_equal(correction[rows], predict_copula(loaded, x))
            replay[f"x_{spec['id']}"] = x
            replay[f"expected_{spec['id']}"] = correction[rows]
            learned[spec["id"]] = loaded
            fits.append({"fold": spec["id"], "train_rows": int(train.sum()), "validation_rows": int(validation.sum()), "unique_fits": 1, "seed": None, "deterministic": True, "covariance_shrinkage_train_only": float(model["shrinkage"]), "fit_seconds": fit_time, "model_path": str(model_file.relative_to(ROOT)), "model_sha256": base.file_hash(model_file)})
            base.atomic_json(OUT / "progress.json", {"status": "RUNNING", "pid": os.getpid(), "completed_fits": len(fits), "runtime_seconds": time.monotonic() - started})
        assert np.array_equal(c, old["C_mean"])
        intact = policies(c, correction, cfg)
        natural = ~np.isfinite(query[["temp_5", "psal_5"]].to_numpy(float)).all(axis=1)
        scopes = {"pooled": np.ones(len(truth), bool), "autumn_primary": fold == cfg["primary_fold"], "natural_missing": natural, "natural_present": ~natural, **{name: fold == name for name in np.unique(fold)}}
        panels = metric_table(truth, intact, scopes)
        outputs = {"key": key, "truth": truth, "fold": fold, "natural_missing": natural, **{f"intact_{arm}": pred for arm, pred in intact.items()}}
        episodes = []
        for episode in json.loads((ROOT / cfg["episodes_source"]).read_text(encoding="utf-8"))["episodes"]:
            altered, selected = episode_frame(query, episode)
            support = np.isfinite(altered.baseline) & (altered.public_temp_count >= 2)
            summary = {**episode, "all_keys": len(truth), "injected_rows": int(selected.sum()), "unsupported_rows": int((~support).sum()), "evaluation_rows_deleted": 0}
            outputs[f"{episode['id']}_selected"] = selected
            if not support.all():
                summary["status"] = "SUPPORT_BLOCKED"
                episodes.append(summary)
                continue
            rows = fold == episode["fold"]
            local = altered.loc[rows].reset_index(drop=True)
            changed_c, changed_r = c.copy(), correction.copy()
            changed_c[rows] = cmean(local, episode["fold"])
            changed_r[rows] = predict_copula(learned[episode["fold"]], physical_features(local, changed_c[rows]))
            predictions = policies(changed_c, changed_r, cfg)
            assert all(np.array_equal(pred[~selected], intact[name][~selected]) for name, pred in predictions.items())
            summary.update(status="SCORED", natural_noop=bool(np.array_equal(changed_c, c) and np.array_equal(changed_r, correction)), metrics=metric_table(truth, predictions, {"episode": selected, "outside_episode": ~selected}))
            outputs.update({f"{episode['id']}_{arm}": value for arm, value in predictions.items()})
            episodes.append(summary)
        primary = panels["autumn_primary"]["metrics"]
        decision = "full" if primary["full"]["rmse"] < primary["C"]["rmse"] else "half" if primary["half"]["rmse"] < primary["C"]["rmse"] else "C"
        np.savez_compressed(OUT / "historical_eval.npz", **outputs)
        np.savez_compressed(OUT / "replay_input.npz", **replay)
        result = {"experiment_id": ID, "status": "COMPLETE_EXPLORATORY_INTERNAL_ONLY", "selected_predefined_policy": decision, "new_historical_fits": len(fits), "new_backbone_fits": 0, "new_full_fits": 0, "reused_C_historical_fits": 9, "fit_receipts": fits, "intact": panels, "episodes": episodes, "runtime_seconds": time.monotonic() - started, "runner_sha256": base.file_hash(Path(__file__)), "config_sha256": base.file_hash(CONFIG), "raw_sha256": base.file_hash(OUT / "historical_eval.npz"), "replay_input_sha256": base.file_hash(OUT / "replay_input.npz"), "official_access_rows": 0, "csv_written": 0, "upload": 0, "evaluation_rows_deleted": 0, "gpu_used": False, "baseline_empty_model_training_reproduction": "NOT_RUN_ROOT_COORDINATES_SEPARATE_SCRATCH_TEST", "limitations": ["Outer-training residual uses in-sample C3 baseline predictions; not inner cross-fitted", "Repeated historical panels: exploratory evidence, not fresh confirmation", "All scores use released finite-label proxy; hidden QC unavailable", "Missing latent covariance uses zero rank imputation; missing conditioners marginalized at prediction", "Four artificial autumn3d unsupported rows block the whole scenario; not an official data defect", "30 minutes is an operational plan, not a runner hard timeout", "Historical C9 exact reload is not blank-model scratch reproduction"]}
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", result)
        base.atomic_json(OUT / "progress.json", {"status": "COMPLETE", "completed_fits": len(fits), "runtime_seconds": result["runtime_seconds"]})
        print(json.dumps({"status": result["status"], "policy": decision, "new_fits": len(fits), "runtime_seconds": result["runtime_seconds"]}))
    except Exception as exc:
        save(OUT / "terminal_failure.json", {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "message": str(exc), "completed_fits": len(fits), "runtime_seconds": time.monotonic() - started})
        raise


def replay():
    cfg = read_config()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))["hashes"]
    assert seal == fingerprint()
    install_guard(seal)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    lock = json.loads((OUT / "ATTEMPT_LOCK.json").read_text(encoding="utf-8"))
    assert os.getpid() != lock["pid"]
    assert base.file_hash(OUT / "replay_input.npz") == result["replay_input_sha256"]
    inputs = np.load(OUT / "replay_input.npz", allow_pickle=False)
    rows, largest = 0, 0.0
    for receipt in result["fit_receipts"]:
        path = ROOT / receipt["model_path"]
        assert base.file_hash(path) == receipt["model_sha256"]
        model = dict(np.load(path, allow_pickle=False))
        current = predict_copula(model, inputs[f"x_{receipt['fold']}"])
        expected = inputs[f"expected_{receipt['fold']}"]
        assert np.array_equal(current, expected)
        rows += len(current)
        largest = max(largest, float(np.max(np.abs(current - expected))))
    save(REPORT / "replay.json", {"status": "PASS", "pid": os.getpid(), "training_pid": lock["pid"], "rows": rows, "maximum_absolute_error": largest, "scope": "all_69850_historical_copula_corrections_saved_features_only_not_official_pipeline", "new_fits": 0, "official_access_rows": cfg["official_access_rows"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "execute", "replay"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.mode == "seal":
            save(SEAL, {"experiment_id": ID, "sealed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hashes": fingerprint()})
            print("SEALED")
        elif args.mode == "execute":
            execute()
        else:
            replay()

"""New-ID numeric amendment with proved exact reuse, never restart the v1 attempt."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_p2_crossfit_copula_forward_20260906_v1 as old
from scipy.special import ndtri

ROOT = old.ROOT
ID = "p2_crossfit_copula_forward_numeric_20260906_v2"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
SEAL = REPORT / "preregistration-seal.json"
base, profile, cv, torch = old.base, old.profile, old.cv, old.torch
threadpool_limits = old.threadpool_limits
save, key_array, key_hash = old.save, old.key_array, old.key_hash
adapter, masks, outage_frame = old.adapter, old.masks, old.outage_frame
load_population, cmean, metric, panels = old.load_population, old.cmean, old.metric, old.panels


def settings():
    amendment = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg, contract, recipe = old.settings()
    assert amendment["experiment_id"] == ID
    assert amendment["new_backbone_fits"] == 36 and amendment["new_copula_fits"] == 12
    assert amendment["reused_backbone_fits"] == 36 and amendment["reused_copula_fit_saves"] == 4
    assert amendment["correction_strength"] == 1.0 and not amendment["gpu_used"]
    assert amendment["cpu_threads"] == 2 and amendment["outside_outage_output_rtol"] == 0.0
    assert amendment["outside_outage_output_atol_celsius"] == 1e-12
    cfg.update(amendment)
    return cfg, contract, recipe


def fingerprints():
    return {"runner": base.file_hash(Path(__file__)), "config": base.file_hash(CONFIG),
            "original": old.fingerprints()}


def inventory():
    """Hash only known parent outputs; no other experiment models or answers."""
    paths = [old.SEAL, old.CONFIG, old.OUT / "ATTEMPT_LOCK.json",
             old.OUT / "terminal_failure.json", old.OUT / "fit-receipts.json",
             old.OUT / "baseline_oof.npz", old.REPORT / "baseline-result.json",
             old.REPORT / "resource-receipt.json", old.REPORT / "technical-diagnostic.json"]
    paths += list((old.OUT / "03_model").glob("*.pt"))
    paths += list((old.OUT / "03_model").glob("*.npz"))
    paths += list(old.OUT.glob("calibration_B[12]_*.npz"))
    paths += list(old.OUT.glob("replay_B[12]_*.npz"))
    return {path.relative_to(ROOT).as_posix(): base.file_hash(path) for path in sorted(paths)}


def install_guard():
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    allowed_old = {(ROOT / name).resolve() for name in sealed["parent_inventory"]}

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("network prohibited")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1]
        writing = (isinstance(mode, str) and any(v in mode for v in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR)))
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden access prohibited")
        if path.is_relative_to(old.OUT.resolve()) or path.is_relative_to(old.REPORT.resolve()):
            if writing:
                raise PermissionError("parent attempt is immutable")
        if path == source and writing:
            raise PermissionError("source immutable")
        if path.suffix.lower() == ".csv" and path != source:
            raise PermissionError("only distributed observations CSV authorized")
        if path.suffix.lower() in {".pt", ".pth", ".npz", ".npy", ".parquet", ".cbm", ".joblib"}:
            if not path.is_relative_to(OUT.resolve()) and not (not writing and path in allowed_old):
                raise PermissionError("unapproved model or array")

    sys.addaudithook(guard)
    return source


def assert_frozen():
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert sealed["hashes"] == fingerprints()
    assert sealed["parent_inventory"] == inventory()
    parent_seal = json.loads(old.SEAL.read_text(encoding="utf-8"))
    assert parent_seal["hashes"] == old.fingerprints()
    return sealed


def load_models(fold, stage, seeds):
    models = []
    for seed in seeds:
        model = base.make_model("v23", 11)
        model.load_state_dict(torch.load(OUT / "03_model" / f"{fold}_{stage}_{seed}.pt",
                                        map_location="cpu", weights_only=True))
        models.append(model.eval())
    return models


def verify_numeric_invariance(keys, altered_keys, c, ac, nx, ax, selected, natural, altered):
    """Amend output roundoff only; never modify the supplied predictions."""
    outside = ~selected
    assert np.array_equal(keys, altered_keys), "key/order mismatch"
    assert np.array_equal(c[outside], ac[outside]), "baseline changed outside outage"
    assert np.array_equal(nx[outside], ax[outside], equal_nan=True), "physical inputs changed"
    assert np.isfinite(natural).all() and np.isfinite(altered).all()
    delta = np.abs(natural[outside] - altered[outside])
    assert np.all(delta <= 1e-12), "outside-outage output exceeds predeclared roundoff"
    return {"outside_rows": int(outside.sum()), "input_key_baseline_features_exact": True,
            "output_different_rows": int(np.count_nonzero(delta)),
            "max_abs_output_difference_celsius": float(delta.max()) if len(delta) else 0.0,
            "output_rtol": 0.0, "output_atol_celsius": 1e-12, "prediction_values_modified": False}


def covariance_checks(model, x, residual):
    """Independent analytic Ledoit-Wolf/CDF verification; no estimator.fit call."""
    checks = {}
    zcols = []
    for index, col in enumerate([*x.T, residual]):
        marginal = np.sort(col[np.isfinite(col)])
        saved = model[f"marginal_{index}"] if index < 11 else model["response"]
        checks[f"cdf_{index}"] = bool(np.array_equal(marginal, saved))
        latent = np.zeros(len(col))
        available = np.isfinite(col)
        if len(marginal):
            probability = (np.searchsorted(marginal, col[available], "left")
                           + np.searchsorted(marginal, col[available], "right")) / (2 * len(marginal))
            latent[available] = ndtri(np.clip(probability, .5 / len(marginal), 1 - .5 / len(marginal)))
        zcols.append(latent)
    z = np.column_stack(zcols)
    centered = z - z.mean(axis=0)
    n, p = centered.shape
    gram = centered.T @ centered
    empirical = gram / n
    trace = np.sum(centered ** 2, axis=0) / n
    mu = float(trace.sum() / p)
    delta_raw = float(np.sum(gram ** 2) / n ** 2)
    squared = centered ** 2
    beta = float((np.sum(squared.T @ squared) / n - delta_raw) / (p * n))
    delta = float((delta_raw - 2 * mu * trace.sum() + p * mu ** 2) / p)
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(p)
    checks["location"] = bool(np.allclose(model["location"], z.mean(axis=0), rtol=0, atol=1e-12))
    checks["shrinkage"] = bool(np.isclose(model["shrinkage"], shrinkage, rtol=1e-10, atol=1e-12))
    checks["covariance"] = bool(np.allclose(model["covariance"], covariance, rtol=1e-10, atol=1e-12))
    assert all(checks.values()), "copula CDF/covariance provenance mismatch"
    return checks


def calibration(frame, truth, spec, support, cfg):
    train, valid = masks(frame.time, spec)
    local = frame.loc[train].reset_index(drop=True)
    c = cmean(load_models(spec["id"], "outer", cfg["seeds"]), local)
    result = {"insample_full": (np.flatnonzero(train), profile.physical_features(local, c), truth[train] - c)}
    indices, features, residuals = [], [], []
    for inner in support["inner"]:
        _, iv = masks(frame.time, inner)
        assert np.all(train[iv]) and not np.any(valid & iv)
        local = frame.loc[iv].reset_index(drop=True)
        c = cmean(load_models(spec["id"], inner["id"], cfg["seeds"]), local)
        indices.extend(np.flatnonzero(iv).tolist())
        features.append(profile.physical_features(local, c))
        residuals.append(truth[iv] - c)
    result["crossfit_full"] = (np.asarray(indices), np.concatenate(features), np.concatenate(residuals))
    return result


def zero_fit_reuse():
    cfg, contract, recipe = settings()
    sealed = assert_frozen()
    if OUT.exists():
        raise FileExistsError("new exactly-once reuse directory required")
    source = install_guard()
    started = time.monotonic()
    (OUT / "03_model").mkdir(parents=True)
    save(OUT / "REUSE_LOCK.json", {"pid": os.getpid(), "seal_sha256": base.file_hash(SEAL)})
    checks, copulas = {}, []

    def check(name, condition):
        checks[name] = bool(condition)
        assert checks[name], name

    try:
        frame, truth = load_population(cfg)
        support = old.support_audit(frame, truth, cfg, contract)
        check("source_support_and_inner_months_exact", support == sealed["support"])
        resource = json.loads((old.REPORT / "resource-receipt.json").read_text())
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "torch_installed": torch.__version__,
                    "scipy": importlib.metadata.version("scipy"),
                    "sklearn": importlib.metadata.version("scikit-learn")}
        check("runtime_versions_exact", all(resource[k] == v for k, v in versions.items()))
        prior_fits = json.loads((old.OUT / "fit-receipts.json").read_text())
        check("36_distinct_backbone_receipts", len(prior_fits) == len({r["fit_id"] for r in prior_fits}) == 36)
        for path in sorted((old.OUT / "03_model").iterdir()):
            if path.suffix in {".pt", ".npz"}:
                shutil.copyfile(path, OUT / "03_model" / path.name)
        check("36_backbone_4_copula_saved", len(list((OUT / "03_model").glob("*.pt"))) == 36
              and len(list((OUT / "03_model").glob("*.npz"))) == 4)
        for path in sorted(old.OUT.glob("calibration_B[12]_*.npz")):
            shutil.copyfile(path, OUT / path.name)
        shutil.copyfile(old.OUT / "baseline_oof.npz", OUT / "baseline_oof.npz")
        shutil.copyfile(old.REPORT / "baseline-result.json", REPORT / "baseline-result.json")
        baseline = np.load(OUT / "baseline_oof.npz", allow_pickle=False)
        check("same_keys_truth", np.array_equal(key_array(frame), baseline["key"]) and np.array_equal(truth, baseline["truth"]))
        for spec in contract["P2"]["folds"]:
            train, valid = masks(frame.time, spec)
            row = next(s for s in support["folds"] if s["fold"] == spec["id"])
            stages = {"outer": train}
            if spec["id"] in {"B1", "B2"}:
                for inner in row["inner"]:
                    it, iv = masks(frame.time, inner)
                    check(f"{spec['id']}/{inner['id']}/double_purge", not np.any(valid & (train & it)) and np.all(train[iv]))
                    stages[inner["id"]] = train & it
            for stage, selected in stages.items():
                _, training = base.training_arrays(frame.loc[selected].reset_index(drop=True), truth[selected], "v23_blockmask", recipe)
                for seed in cfg["seeds"]:
                    identifier = f"{spec['id']}_{stage}_{seed}"
                    fit = next(r for r in prior_fits if r["fit_id"] == identifier)
                    check(identifier + "/recipe", fit["seed"] == seed and fit["epochs"] == 60 and fit["device"] == "cpu" and fit["cpu_threads"] == 2)
                    check(identifier + "/training", fit["training"] == training and fit["train_key_sha256"] == key_hash(frame.loc[selected]))
                    check(identifier + "/model", base.file_hash(OUT / "03_model" / f"{identifier}.pt") == fit["model_sha256"])
            query = frame.loc[valid].reset_index(drop=True)
            altered, selected = outage_frame(query, spec)
            models = load_models(spec["id"], "outer", cfg["seeds"])
            check(spec["id"] + "/baseline_natural_exact", np.array_equal(cmean(models, query), baseline["natural_C3"][valid]))
            check(spec["id"] + "/baseline_outage_exact", np.array_equal(cmean(models, altered), baseline["outage_C3"][valid]))
            if spec["id"] not in {"B1", "B2"}:
                continue
            for arm, (indices, x, residual) in calibration(frame, truth, spec, row, cfg).items():
                path = OUT / "03_model" / f"{spec['id']}_{arm}.npz"
                data_path = OUT / f"calibration_{spec['id']}_{arm}.npz"
                fitted, data = dict(np.load(path, allow_pickle=False)), np.load(data_path, allow_pickle=False)
                check(f"{spec['id']}/{arm}/calibration_exact", np.array_equal(indices, data["row_indices"])
                      and np.array_equal(x, data["physical_x"], equal_nan=True) and np.array_equal(residual, data["residual"]))
                analytic = covariance_checks(fitted, x, residual)
                checks.update({f"{spec['id']}/{arm}/{k}": v for k, v in analytic.items()})
                c, ac = baseline["natural_C3"][valid], baseline["outage_C3"][valid]
                nx, ax = profile.physical_features(query, c), profile.physical_features(altered, ac)
                natural, outage = c + profile.predict_copula(fitted, nx), ac + profile.predict_copula(fitted, ax)
                invariant = verify_numeric_invariance(key_array(query), key_array(altered), c, ac, nx, ax, selected, natural, outage)
                same_shape = np.array_equal(profile.predict_copula(fitted, nx[~selected]), profile.predict_copula(fitted, ax[~selected]))
                check(f"{spec['id']}/{arm}/same_shape_exact", same_shape)
                prior_replay = old.OUT / f"replay_{spec['id']}_{arm}.npz"
                if prior_replay.exists():
                    previous = np.load(prior_replay, allow_pickle=False)
                    check(f"{spec['id']}/{arm}/prior_output_exact", np.array_equal(nx, previous["natural_x"], equal_nan=True)
                          and np.array_equal(ax, previous["outage_x"], equal_nan=True)
                          and np.array_equal(natural - c, previous["natural_correction"])
                          and np.array_equal(outage - ac, previous["outage_correction"]))
                copulas.append({"fold": spec["id"], "arm": arm, "reused": True,
                    "calibration_rows": len(residual), "runtime_seconds": None,
                    "runtime_note": "Parent fit succeeded/saved; per-copula runtime absent after terminal failure",
                    "model_sha256": base.file_hash(path), "calibration_sha256": base.file_hash(data_path),
                    "outer_labels_used": 0, "outer_trained_label_models_used_in_crossfit": 0,
                    "inner_key_sha256": key_hash(frame.iloc[indices]) if arm == "crossfit_full" else None,
                    "missing_post_assert_receipt_reconstructed": not prior_replay.exists(),
                    "numeric_invariance": invariant, "analytic_checks": analytic})
                np.savez_compressed(OUT / f"reuse_prediction_{spec['id']}_{arm}.npz", natural=natural, outage=outage)
        check("source_unchanged", base.file_hash(source) == cfg["source_sha256"])
        assert_frozen()
        save(OUT / "reused-fit-receipts.json", prior_fits)
        save(OUT / "reused-copula-receipts.json", copulas)
        payload = {"status": "ZERO_FIT_REUSE_PROVENANCE_PASS", "pid": os.getpid(), "checks": checks,
                   "checks_count": len(checks), "reused_backbones": 36, "copula_fit_saves": 4,
                   "parent_post_assert_copula_receipts": 3, "new_fits": 0,
                   "runtime_seconds": time.monotonic() - started, "performance_metrics_read": 0,
                   "versions": versions, "official_access_rows": 0, "csv_written": 0, "upload": 0,
                   "seal_sha256": base.file_hash(SEAL), "copula_receipts": copulas}
        save(REPORT / "zero-fit-reuse.json", payload)
        print(json.dumps({"status": payload["status"], "checks": len(checks), "seconds": payload["runtime_seconds"]}), flush=True)
    except Exception as exc:
        save(REPORT / "zero-fit-failure.json", {"status": "PROVENANCE_OR_TECHNICAL_BLOCKER", "type": type(exc).__name__,
             "message": str(exc), "new_fits": 0, "checks": checks, "automatic_restart": False})
        raise


def execute():
    cfg, contract, recipe = settings()
    sealed = assert_frozen()
    verified = json.loads((REPORT / "zero-fit-reuse.json").read_text())
    assert verified["status"] == "ZERO_FIT_REUSE_PROVENANCE_PASS" and verified["new_fits"] == 0
    source = install_guard()
    save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "seal_sha256": base.file_hash(SEAL),
         "zero_fit_sha256": base.file_hash(REPORT / "zero-fit-reuse.json"), "new_fits_cap": 48})
    start = time.monotonic()
    deadline = start + cfg["remaining_execution_cap_seconds"]
    fits = json.loads((OUT / "reused-fit-receipts.json").read_text())
    copulas = json.loads((OUT / "reused-copula-receipts.json").read_text())

    def progress(stage, **extra):
        base.atomic_json(OUT / "progress.json", {"status": "RUNNING", "stage": stage,
            "pid": os.getpid(), "new_backbone_fits": len(fits) - 36, "new_copula_fits": len(copulas) - 4,
            "reused_fits": 40, "runtime_seconds": time.monotonic() - start, **extra})

    def fit_set(selected, frame, truth, fold, stage):
        data, training = base.training_arrays(frame.loc[selected].reset_index(drop=True), truth[selected], "v23_blockmask", recipe)
        for seed in cfg["seeds"]:
            if len(fits) >= 72 or time.monotonic() >= deadline:
                raise old.BudgetStop("remaining original budget reached; no restart")
            identifier = f"{fold}_{stage}_{seed}"
            path = OUT / "03_model" / f"{identifier}.pt"
            if path.exists():
                raise FileExistsError("never refit or overwrite an existing model")
            model, measured = old.fit_cpu(data, seed, recipe,
                lambda epoch, seconds, fit_id=identifier: progress("inner_fit", fit_id=fit_id, epoch=epoch, fit_seconds=seconds), deadline)
            torch.save(model.state_dict(), path)
            reload = base.make_model("v23", 11)
            reload.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            probe = frame.loc[selected].iloc[:128]
            assert np.array_equal(cmean([model], probe), cmean([reload.eval()], probe))
            fits.append({"fit_id": identifier, "fold": fold, "stage": stage,
                "training": training, "train_key_sha256": key_hash(frame.loc[selected]),
                "model_sha256": base.file_hash(path), "reload_128_train_rows": True, "reused": False, **measured})
            base.atomic_json(OUT / "fit-receipts.json", fits)
            progress("inner_fit_complete", fit_id=identifier)

    try:
        frame, truth = load_population(cfg)
        assert old.support_audit(frame, truth, cfg, contract) == sealed["support"]
        raw = np.load(OUT / "baseline_oof.npz", allow_pickle=False)
        assert np.array_equal(key_array(frame), raw["key"]) and np.array_equal(truth, raw["truth"])
        folds, outaged = raw["fold"], raw["outage_mask"]
        natural = {arm: raw["natural_C3"].copy() for arm in cfg["policies"]}
        outage = {arm: raw["outage_C3"].copy() for arm in cfg["policies"]}
        for spec in contract["P2"]["folds"]:
            train, valid = masks(frame.time, spec)
            support = next(row for row in sealed["support"]["folds"] if row["fold"] == spec["id"])
            if spec["id"] in {"B1", "B2"}:
                for arm in ("insample_full", "crossfit_full"):
                    reused = np.load(OUT / f"reuse_prediction_{spec['id']}_{arm}.npz", allow_pickle=False)
                    natural[arm][valid], outage[arm][valid] = reused["natural"], reused["outage"]
                continue
            for inner in support["inner"]:
                it, iv = masks(frame.time, inner)
                assert np.all(train[iv]) and not np.any(valid & (train & it))
                fit_set(train & it, frame, truth, spec["id"], inner["id"])
            query = frame.loc[valid].reset_index(drop=True)
            altered, selected = outage_frame(query, spec)
            for arm, (indices, x, residual) in calibration(frame, truth, spec, support, cfg).items():
                if time.monotonic() >= deadline or len(copulas) >= 16:
                    raise old.BudgetStop("remaining budget before next scheduled copula")
                before = time.monotonic()
                path = OUT / "03_model" / f"{spec['id']}_{arm}.npz"
                data_path = OUT / f"calibration_{spec['id']}_{arm}.npz"
                if path.exists() or data_path.exists():
                    raise FileExistsError("never overwrite prior copula/calibration")
                np.savez_compressed(data_path, row_indices=indices, physical_x=x, residual=residual)
                fitted = profile.fit_copula(x, residual)
                np.savez_compressed(path, **fitted)
                c, ac = natural["C3"][valid], outage["C3"][valid]
                nx, ax = profile.physical_features(query, c), profile.physical_features(altered, ac)
                natural[arm][valid] = c + profile.predict_copula(fitted, nx)
                outage[arm][valid] = ac + profile.predict_copula(fitted, ax)
                invariant = verify_numeric_invariance(key_array(query), key_array(altered), c, ac, nx, ax,
                    selected, natural[arm][valid], outage[arm][valid])
                copulas.append({"fold": spec["id"], "arm": arm, "reused": False,
                    "calibration_rows": len(residual), "runtime_seconds": time.monotonic() - before,
                    "model_sha256": base.file_hash(path), "calibration_sha256": base.file_hash(data_path),
                    "outer_labels_used": 0, "outer_trained_label_models_used_in_crossfit": 0,
                    "inner_key_sha256": key_hash(frame.iloc[indices]) if arm == "crossfit_full" else None,
                    "numeric_invariance": invariant})
                base.atomic_json(OUT / "copula-receipts.json", copulas)
            progress("candidate_fold_complete", fold=spec["id"])
        assert len(fits) == 72 and len(copulas) == 16
        comparisons = {}
        for scope, selected in (("primary_B3", folds == "B3"), ("pooled", np.ones(len(frame), bool)), ("outage_interval", outaged)):
            groups = cv.bootstrap_groups(frame.loc[selected], "P2", contract)
            predictions = outage if scope == "outage_interval" else natural
            comparisons[scope] = {arm: cv.paired_bootstrap(truth[selected], predictions["C3"][selected],
                predictions[arm][selected], groups, "rmse", contract) for arm in ("insample_full", "crossfit_full")}
            comparisons[scope]["crossfit_vs_insample"] = cv.paired_bootstrap(truth[selected], predictions["insample_full"][selected],
                predictions["crossfit_full"][selected], groups, "rmse", contract)
        path = OUT / "evaluation.npz"
        np.savez_compressed(path, key=key_array(frame), truth=truth, fold=folds, layer=frame.layer.to_numpy(),
            time=frame.time.to_numpy(str), natural_T5_missing=~np.isfinite(frame.temp_5.to_numpy(float)), outage_mask=outaged,
            **{f"natural_{k}": v for k, v in natural.items()}, **{f"outage_{k}": v for k, v in outage.items()})
        assert_frozen()
        result = {"experiment_id": ID, "status": "COMPLETE_INTERNAL_8FOLD_CPU_ONLY", "training_pid": os.getpid(),
            "baseline_result_sha256": base.file_hash(REPORT / "baseline-result.json"), "support": sealed["support"],
            "new_backbone_fits": 36, "new_copula_fits": 12, "reused_backbone_fits": 36, "reused_copula_fits": 4,
            "logical_backbone_fits": 72, "logical_copula_fits": 16, "new_full_fits": 0,
            "fit_receipts": fits, "copula_receipts": copulas,
            "natural": panels(frame, truth, natural, folds, outaged, contract),
            "testmatched": panels(frame, truth, outage, folds, outaged, contract), "comparisons": comparisons,
            "retained_mean_improvement_candidates": [arm for arm in ("insample_full", "crossfit_full") if comparisons["primary_B3"][arm]["candidate_retained"]],
            "automatic_promotion": False, "runtime_seconds": time.monotonic() - start,
            "previous_execution_seconds": cfg["previous_execution_seconds"],
            "combined_execution_seconds": time.monotonic() - start + cfg["previous_execution_seconds"],
            "source_sha256_unchanged": base.file_hash(source) == cfg["source_sha256"], "evaluation_sha256": base.file_hash(path),
            "fit_receipts_sha256": base.file_hash(OUT / "fit-receipts.json"), "runner_sha256": sealed["hashes"]["runner"],
            "config_sha256": sealed["hashes"]["config"], "official_access_rows": 0, "csv_written": 0, "upload": 0,
            "evaluation_rows_deleted": 0, "gpu_used": False, "limitations": cfg["limitations"]}
        assert result["combined_execution_seconds"] <= 5400, "combined research execution cap exceeded"
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", result)
        progress("COMPLETE")
        print(json.dumps({"status": result["status"], "new_fits": 48, "reused_fits": 40, "seconds": result["runtime_seconds"]}), flush=True)
    except Exception as exc:
        failure = {"experiment_id": ID, "status": "TERMINAL_RESOURCE_BUDGET_LIMIT" if isinstance(exc, old.BudgetStop) else "TERMINAL_TECHNICAL_FAILURE",
            "exception_type": type(exc).__name__, "message": str(exc), "new_backbone_fits": len(fits) - 36,
            "new_copula_fits": len(copulas) - 4, "runtime_seconds": time.monotonic() - start,
            "automatic_restart": False, "scientific_decision": "NOT_ESTIMATED_INCOMPLETE", "official_access_rows": 0, "csv_written": 0, "upload": 0}
        save(REPORT / "terminal-failure.json", failure)
        save(OUT / "terminal_failure.json", failure)
        raise


def replay():
    cfg, contract, _ = settings()
    assert_frozen()
    source = install_guard()
    result = json.loads((REPORT / "result.json").read_text())
    assert result["training_pid"] != os.getpid()
    raw = np.load(OUT / "evaluation.npz", allow_pickle=False)
    assert base.file_hash(OUT / "evaluation.npz") == result["evaluation_sha256"]
    frame, truth = load_population(cfg)
    assert np.array_equal(key_array(frame), raw["key"]) and np.array_equal(truth, raw["truth"])
    checks = []
    for spec in contract["P2"]["folds"]:
        _, valid = masks(frame.time, spec)
        query = frame.loc[valid].reset_index(drop=True)
        altered, _ = outage_frame(query, spec)
        models = load_models(spec["id"], "outer", cfg["seeds"])
        for surface, local in (("natural", query), ("outage", altered)):
            c = cmean(models, local)
            assert np.array_equal(c, raw[f"{surface}_C3"][valid])
            for arm in ("insample_full", "crossfit_full"):
                model = dict(np.load(OUT / "03_model" / f"{spec['id']}_{arm}.npz", allow_pickle=False))
                prediction = c + profile.predict_copula(model, profile.physical_features(local, c))
                expected = raw[f"{surface}_{arm}"][valid]
                assert np.array_equal(prediction, expected), "same-batch fresh-process replay must be exact"
                checks.append({"fold": spec["id"], "surface": surface, "arm": arm, "n": len(local), "exact": True, "max_error": 0.0})
    assert_frozen()
    assert base.file_hash(source) == cfg["source_sha256"]
    payload = {"status": "PASS", "pid": os.getpid(), "training_pid": result["training_pid"], "rows_per_surface": len(frame),
        "checks": checks, "max_error": 0.0, "new_fits": 0, "official_access_rows": 0, "csv_written": 0, "upload": 0,
        "scope": "fresh_process_all_8fold_3arms_2surfaces_exact_not_official"}
    save(REPORT / "replay.json", payload)
    print(json.dumps({"status": "REPLAY_PASS", "checks": len(checks), "rows": len(frame)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "reuse", "execute", "replay"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "seal":
            cfg, _, _ = settings()
            original = json.loads(old.SEAL.read_text())
            assert old.fingerprints() == original["hashes"]
            save(SEAL, {"experiment_id": ID, "hashes": fingerprints(), "parent_inventory": inventory(),
                "support": original["support"], "source_sha256": cfg["source_sha256"],
                "sealed_at": pd.Timestamp.now(tz="UTC").isoformat()})
            print("SEALED")
        elif args.mode == "reuse":
            zero_fit_reuse()
        elif args.mode == "execute":
            execute()
        else:
            replay()


if __name__ == "__main__":
    main()

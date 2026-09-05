"""Frozen CPU C3 baseline and inner-purged-OOF profile copula comparison."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np  # noqa: E402
import ocean_evaluation_contract_v5 as cv  # noqa: E402
import pandas as pd  # noqa: E402
import run_p2_profile_copula_residual_20260905_v4 as profile  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from torch.nn import functional as F  # noqa: E402

base = profile.base
ROOT = Path(__file__).resolve().parents[1]
ID = "p2_crossfit_copula_forward_20260906_v1"
CONFIG = ROOT / "configs/experiments" / f"{ID}.json"
OUT, REPORT = ROOT / "artifacts" / ID, ROOT / "reports" / ID
SEAL = REPORT / "preregistration-seal.json"
KEYS = ["station", "layer", "time"]


class BudgetStop(RuntimeError):
    """Resource terminal, never a scientific no-go."""


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def settings():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == ID and cfg["seeds"] == [20260901, 20260902, 20260903]
    assert cfg["epochs"] == 60 and cfg["cpu_threads"] == 2 and not cfg["gpu_used"]
    assert cfg["maximum_backbone_fits"] == 72 and cfg["maximum_copula_fits"] == 16
    assert cfg["correction_strength"] == 1.0 and cfg["physical_features"] == 11
    contract = cv.load_contract(ROOT / cfg["evaluation_contract"])
    recipe = json.loads((ROOT / cfg["recipe"]).read_text(encoding="utf-8"))
    recipe["cpu_threads"] = 2
    return cfg, contract, recipe


def fingerprints():
    cfg, _, _ = settings()
    dependencies = [
        "scripts/run_p2_profile_copula_residual_20260905_v4.py",
        "scripts/run_p2_objective_alignment_20260905_v2.py",
        "scripts/run_p2_score_repair_20260905_v1.py",
        "scripts/ocean_evaluation_contract_v5.py",
        *base.DEPENDENCIES,
        cfg["recipe"],
        cfg["evaluation_contract"],
        cfg["source_support_receipt"],
    ]
    return {
        "runner": base.file_hash(Path(__file__)),
        "config": base.file_hash(CONFIG),
        "dependencies": {p: base.file_hash(ROOT / p) for p in dependencies},
    }


def install_guard():
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()

    def hook(event, args):
        if event == "socket.connect":
            raise PermissionError("network prohibited")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1]
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR))
        )
        if "external_data" in path.parts or "hidden" in path.name.lower():
            raise PermissionError("external/hidden path prohibited")
        if path == source and write:
            raise PermissionError("source is immutable")
        if path.suffix.lower() == ".csv" and path != source:
            raise PermissionError("only released observations.csv permitted")
        if (
            path.suffix.lower() in {".pt", ".pth", ".npz", ".npy", ".parquet", ".cbm"}
            and OUT.resolve() not in path.parents
        ):
            raise PermissionError("old models/arrays prohibited")

    sys.addaudithook(hook)
    return source


def key_array(frame):
    return (
        frame.station.astype(str) + "|" + frame.layer.astype(str) + "|" + frame.time.astype(str)
    ).to_numpy(str)


def key_hash(frame):
    return hashlib.sha256("\n".join(key_array(frame)).encode()).hexdigest()


def adapter(observations):
    """Labels separated before all target T/S context is blanked."""
    selected = observations.layer.isin([2, 3, 4])
    truth = observations.loc[selected, "temp"].to_numpy(float).copy()
    masked = observations.copy()
    masked.loc[selected, ["temp", "psal"]] = np.nan
    frame, empty_labels = base.public_frame(masked)
    assert np.isnan(empty_labels).all()
    assert not {f"{v}_{layer}" for v in ("temp", "psal") for layer in (2, 3, 4)}.intersection(frame)
    return frame, truth


def load_population(cfg):
    source = (Path(os.environ["P2_DATA_DIR"]) / "observations.csv").resolve()
    assert base.file_hash(source) == cfg["source_sha256"]
    obs = pd.read_csv(source)
    assert set(obs.columns) == {
        "station",
        "year",
        "time",
        "layer",
        "depth",
        "nominal_depth",
        "temp",
        "psal",
    }
    assert not obs.duplicated(KEYS).any()
    if any(pd.Timestamp(value).tzinfo is None for value in obs.time.unique()):
        raise ValueError("source timestamps must be aware")
    obs.time = pd.to_datetime(obs.time, utc=True)
    assert set(obs.station) == {"S-ORS"}
    frame, truth = adapter(obs)
    eligible = np.isfinite(truth) & np.isfinite(frame.baseline) & (frame.public_temp_count >= 2)
    frame, truth = frame.loc[eligible].reset_index(drop=True), truth[eligible]
    assert len(frame) == 166268 and not frame.duplicated(KEYS).any()
    return frame, truth


def masks(times, spec, purge=7):
    local = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    left, right = base.utc(spec["start"]), base.utc(spec.get("end", spec.get("stop")))
    return (
        np.asarray(
            (local < left - pd.Timedelta(days=purge)) | (local >= right + pd.Timedelta(days=purge))
        ),
        np.asarray((local >= left) & (local < right)),
    )


def choose_inner(frame, outer):
    outer_train, _ = masks(frame.time, outer)
    left, right = (
        base.utc(outer["start"]) - pd.Timedelta(days=7),
        base.utc(outer["end"]) + pd.Timedelta(days=7),
    )
    months = pd.date_range("2024-05-01", "2026-01-01", freq="MS", tz="Asia/Seoul")
    options = []
    for start, end in zip(months[:-1], months[1:], strict=True):
        if not (end.tz_convert("UTC") <= left or start.tz_convert("UTC") >= right):
            continue
        spec = {"start": start.isoformat(), "end": end.isoformat()}
        inner_train, inner_valid = masks(frame.time, spec)
        if not np.any(inner_valid):
            continue
        assert np.all(outer_train[inner_valid])
        train = outer_train & inner_train
        options.append(
            {**spec, "validation_rows": int(inner_valid.sum()), "train_rows": int(train.sum())}
        )
    positions = [(len(options) - 1) // 3, (2 * (len(options) - 1)) // 3]
    if len(options) < 3 or len(set(positions)) != 2:
        raise ValueError("insufficient distinct inner calendar months")
    return [{"id": f"I{i + 1}", **options[p]} for i, p in enumerate(positions)]


def outage_frame(frame, spec):
    parsed = pd.to_datetime(frame.time, utc=True)
    end = base.utc(spec["end"])
    selected = np.asarray((parsed >= end - pd.Timedelta(days=17)) & (parsed < end))
    altered = frame.copy()
    altered.loc[selected, ["temp_5", "psal_5"]] = np.nan
    altered = base.refresh_public(altered)
    values = altered[[f"temp_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    count = np.isfinite(values).sum(axis=1)
    mean = np.divide(
        np.nansum(values, axis=1), count, out=np.full(len(values), np.nan), where=count > 0
    )
    variance = np.divide(
        np.nansum((values - mean[:, None]) ** 2, axis=1),
        count,
        out=np.full(len(values), np.nan),
        where=count > 0,
    )
    altered["public_temp_mean"], altered["public_temp_std"] = mean, np.sqrt(variance)
    return altered, selected


def support_audit(frame, truth, cfg, contract):
    expected = json.loads((ROOT / cfg["source_support_receipt"]).read_text(encoding="utf-8"))["P2"]
    rows, assignment = [], np.zeros(len(frame), int)
    for spec in contract["P2"]["folds"]:
        split = cv.p2_split(frame, spec["id"], contract, KEYS)
        train, valid = masks(frame.time, spec)
        assert np.array_equal(train, split["train"]) and np.array_equal(valid, split["validation"])
        assignment += valid
        reference = next(row for row in expected["folds"] if row["fold"] == spec["id"])
        assert (int(train.sum()), int(valid.sum())) == (reference["train"], reference["validation"])
        local = frame.loc[valid].reset_index(drop=True)
        tokens, token_mask, context = base.arrays(local)
        actual_count = np.column_stack(
            [
                np.isfinite(local[f"temp_{layer}"])
                & np.isfinite(local[f"depth_{layer}"])
                & local[f"depth_{layer}"].gt(0)
                for layer in base.PUBLIC_LAYERS
            ]
        ).sum(axis=1)
        affected = actual_count < 2
        assert int(affected.sum()) == reference["under_two_temp_positive_depth_rows"]
        actual_x = profile.physical_features(local, local.baseline.to_numpy(float))
        assert np.isfinite(actual_x[affected, :5]).all()
        changed, outage = outage_frame(local, spec)
        assert int(outage.sum()) == reference["outage_target_rows"]
        assert np.isfinite(changed.baseline).all() and (changed.public_temp_count >= 2).all()
        changed_arrays = base.arrays(changed)
        assert all(np.isfinite(v).all() for v in changed_arrays)
        assert all(
            np.array_equal(v[~outage], orig[~outage])
            for v, orig in zip(changed_arrays, (tokens, token_mask, context), strict=True)
        )
        natural = ~np.isfinite(local.temp_5.to_numpy(float))
        assert int(natural.sum()) == reference["natural_missing_T5_target_rows"]
        inner = choose_inner(frame, spec)
        for inner_spec in inner:
            it, iv = masks(frame.time, inner_spec)
            assert not np.any(valid & (train & it)) and not np.any(valid & iv)
            assert np.all(train[iv]) and int((train & it).sum()) == inner_spec["train_rows"]
        rows.append(
            {
                "fold": spec["id"],
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "key_sha256": key_hash(local),
                "outage_rows": int(outage.sum()),
                "outage_status": "SUPPORTED" if outage.any() else "NOT_ESTIMABLE_NO_ROWS",
                "natural_T5_missing_rows": int(natural.sum()),
                "nominal_support_preserved_rows": int(affected.sum()),
                "actual_profile_fallback_finite": True,
                "baseline_token_mask_uses_nominal_not_actual": True,
                "inner": inner,
            }
        )
    assert np.all(assignment == 1) and np.isfinite(truth).all()
    return {
        "status": "FEATURE_ADAPTER_SUPPORT_PASS",
        "eligible_rows": len(frame),
        "key_sha256": key_hash(frame),
        "folds": rows,
        "target_temp_psal_context_masked": True,
        "feature_temporal_dependency_hours": 0,
        "evaluation_rows_deleted": 0,
        "official_access_rows": 0,
        "model_fits": 0,
    }


def fit_cpu(data, seed, recipe, progress, deadline):
    """Canonical raw C recipe, CPU device change only; no old model load."""
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = base.make_model("v23", data[2].shape[1]).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    tensors = tuple(torch.from_numpy(value) for value in data)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = time.monotonic()
    for epoch in range(recipe["epochs"]):
        if time.monotonic() >= deadline:
            raise BudgetStop("90min wall cap reached within fit; partial weights not reused")
        model.train()
        order = torch.randperm(len(data[0]), generator=generator)
        for offset in range(0, len(order), recipe["batch_size"]):
            ids = order[offset : offset + recipe["batch_size"]]
            tokens = tensors[0][ids].detach().clone().requires_grad_(True)
            mask, context, target, weights = [value[ids] for value in tensors[1:]]
            optimizer.zero_grad(set_to_none=True)
            estimate = model(tokens, mask, context)
            losses = F.smooth_l1_loss(estimate, target, beta=1.0, reduction="none")
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
            penalty = base.observed_temperature_gradient_penalty(losses, tokens, mask, weights)
            objective = loss + recipe["gradient_coefficient"] * penalty
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite objective")
            objective.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch + 1 == recipe["epochs"]:
            progress(epoch + 1, time.monotonic() - start)
    return model.eval(), {
        "seed": seed,
        "epochs": recipe["epochs"],
        "runtime_seconds": time.monotonic() - start,
        "device": "cpu",
        "cpu_threads": 2,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def cmean(models, frame):
    return np.mean(
        np.stack([profile.previous.predict_absolute(model, frame) for model in models]), axis=0
    )


def load_models(fold, stage, seeds):
    result = []
    for seed in seeds:
        model = base.make_model("v23", 11)
        model.load_state_dict(
            torch.load(
                OUT / "03_model" / f"{fold}_{stage}_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        result.append(model.eval())
    return result


def metric(y, p):
    if len(y) == 0:
        return {"status": "NOT_ESTIMABLE_NO_ROWS", "n": 0, "sse": None, "rmse": None}
    return {"status": "ESTIMATED", **base.metrics(y, p)}


def panels(frame, truth, prediction, fold, outage, contract):
    scopes = {
        "pooled": np.ones(len(frame), bool),
        "primary_B3": fold == "B3",
        "natural_T5_missing": ~np.isfinite(frame.temp_5.to_numpy(float)),
        "natural_T5_present": np.isfinite(frame.temp_5.to_numpy(float)),
        "outage_interval": outage,
    }
    for value in np.unique(fold):
        scopes[f"fold_{value}"] = fold == value
        scopes[f"outage_{value}"] = (fold == value) & outage
    for layer in (2, 3, 4):
        scopes[f"layer_{layer}"] = frame.layer.to_numpy() == layer
        scopes[f"primary_layer_{layer}"] = (fold == "B3") & (frame.layer.to_numpy() == layer)
    return {
        name: {arm: metric(truth[mask], pred[mask]) for arm, pred in prediction.items()}
        for name, mask in scopes.items()
    }


def execute():
    cfg, contract, recipe = settings()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    assert seal["hashes"] == fingerprints()
    if OUT.exists() or (REPORT / "result.json").exists():
        raise RuntimeError("exactly-once output/terminal exists")
    source = install_guard()
    (OUT / "03_model").mkdir(parents=True)
    save(OUT / "ATTEMPT_LOCK.json", {"pid": os.getpid(), "sealed": seal})
    start, fits, copula = time.monotonic(), [], []
    deadline = start + cfg["max_seconds"]

    def progress(stage, **extra):
        base.atomic_json(
            OUT / "progress.json",
            {
                "status": "RUNNING",
                "stage": stage,
                "pid": os.getpid(),
                "completed_backbone_fits": len(fits),
                "completed_copula_fits": len(copula),
                "runtime_seconds": time.monotonic() - start,
                **extra,
            },
        )

    def fit_set(train, frame, truth, fold_id, stage):
        data, receipt = base.training_arrays(
            frame.loc[train].reset_index(drop=True), truth[train], "v23_blockmask", recipe
        )
        models = []
        for seed in cfg["seeds"]:
            if time.monotonic() >= deadline:
                raise BudgetStop("wall cap before next scheduled fit")
            if len(fits) >= cfg["maximum_backbone_fits"]:
                raise RuntimeError("fit cap")
            identifier = f"{fold_id}_{stage}_{seed}"
            progress("fit", fit_id=identifier)
            model, measured = fit_cpu(
                data,
                seed,
                recipe,
                lambda epoch, seconds, current_id=identifier: progress(
                    "fit", fit_id=current_id, epoch=epoch, fit_seconds=seconds
                ),
                deadline,
            )
            path = OUT / "03_model" / f"{identifier}.pt"
            torch.save(model.state_dict(), path)
            replay_model = base.make_model("v23", 11)
            replay_model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            probe = frame.loc[train].iloc[:128]
            assert np.array_equal(
                profile.previous.predict_absolute(model, probe),
                profile.previous.predict_absolute(replay_model.eval(), probe),
            )
            fits.append(
                {
                    "fit_id": identifier,
                    "fold": fold_id,
                    "stage": stage,
                    "training": receipt,
                    "train_key_sha256": key_hash(frame.loc[train]),
                    "model_sha256": base.file_hash(path),
                    "reload_128_train_rows": True,
                    **measured,
                }
            )
            base.atomic_json(OUT / "fit-receipts.json", fits)
            if len(fits) == 1:
                total_training_rows = sum(
                    row["train_rows"] * 3 + sum(inner["train_rows"] * 3 for inner in row["inner"])
                    for row in seal["support"]["folds"]
                )
                estimated = (
                    measured["runtime_seconds"]
                    / len(data[0])
                    * total_training_rows
                    * 1.3
                    * cfg["pilot_estimate_safety_factor"]
                    + cfg["pilot_nonfit_allowance_seconds"]
                )
                pilot = {
                    "status": "WITHIN_ESTIMATED_BUDGET"
                    if estimated <= cfg["max_seconds"]
                    else "BUDGET_ESTIMATE_EXCEEDS_CAP",
                    "fit_id": identifier,
                    "first_fit_seconds": measured["runtime_seconds"],
                    "estimated_total_seconds": estimated,
                    "cap_seconds": cfg["max_seconds"],
                    "evaluation_predictions_or_metrics_before_decision": 0,
                    "pilot_reused": True,
                    "gpu_used": False,
                }
                save(REPORT / "cpu-pilot.json", pilot)
                print(json.dumps(pilot), flush=True)
                if estimated > cfg["max_seconds"]:
                    raise BudgetStop(
                        "planned CPU workload exceeds 90min before any validation scoring"
                    )
            models.append(model)
        return models

    try:
        frame, truth = load_population(cfg)
        actual_support = support_audit(frame, truth, cfg, contract)
        assert actual_support == seal["support"]
        folds = np.full(len(frame), "", dtype="U2")
        outaged = np.zeros(len(frame), bool)
        natural_c, outage_c = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        for spec in contract["P2"]["folds"]:
            train, valid = masks(frame.time, spec)
            models = fit_set(train, frame, truth, spec["id"], "outer")
            local = frame.loc[valid].reset_index(drop=True)
            altered, selected = outage_frame(local, spec)
            natural_c[valid], outage_c[valid] = cmean(models, local), cmean(models, altered)
            assert np.array_equal(natural_c[valid][~selected], outage_c[valid][~selected])
            folds[valid], outaged[valid] = spec["id"], selected
            progress("baseline_complete_fold", fold=spec["id"])
        assert len(fits) == 24 and np.isfinite(natural_c).all() and np.isfinite(outage_c).all()
        baseline_file = OUT / "baseline_oof.npz"
        np.savez_compressed(
            baseline_file,
            key=key_array(frame),
            truth=truth,
            fold=folds,
            natural_C3=natural_c,
            outage_C3=outage_c,
            outage_mask=outaged,
        )
        baseline = {
            "status": "BASELINE_8FOLD_COMPLETE_FROZEN_BEFORE_CANDIDATES",
            "new_fits": 24,
            "key_sha256": key_hash(frame),
            "oof_sha256": base.file_hash(baseline_file),
            "model_sha256": {r["fit_id"]: r["model_sha256"] for r in fits},
            "natural": panels(frame, truth, {"C3": natural_c}, folds, outaged, contract),
            "testmatched": panels(frame, truth, {"C3": outage_c}, folds, outaged, contract),
            "candidate_fits_before_this_receipt": 0,
        }
        save(REPORT / "baseline-result.json", baseline)
        natural = {
            "C3": natural_c,
            "insample_full": natural_c.copy(),
            "crossfit_full": natural_c.copy(),
        }
        altered_pred = {
            "C3": outage_c,
            "insample_full": outage_c.copy(),
            "crossfit_full": outage_c.copy(),
        }
        for spec in contract["P2"]["folds"]:
            train, valid = masks(frame.time, spec)
            models = load_models(spec["id"], "outer", cfg["seeds"])
            training_frame = frame.loc[train].reset_index(drop=True)
            training_c = cmean(models, training_frame)
            calibration = {
                "insample_full": (
                    profile.physical_features(training_frame, training_c),
                    truth[train] - training_c,
                )
            }
            inner_rows, inner_x, inner_residual = [], [], []
            support_row = next(r for r in seal["support"]["folds"] if r["fold"] == spec["id"])
            for inner in support_row["inner"]:
                inner_train, inner_valid = masks(frame.time, inner)
                selected_train = train & inner_train
                assert not np.any(valid & selected_train) and np.all(train[inner_valid])
                inner_models = fit_set(selected_train, frame, truth, spec["id"], inner["id"])
                local = frame.loc[inner_valid].reset_index(drop=True)
                prediction = cmean(inner_models, local)
                inner_rows.extend(np.flatnonzero(inner_valid).tolist())
                inner_x.append(profile.physical_features(local, prediction))
                inner_residual.append(truth[inner_valid] - prediction)
            assert len(set(inner_rows)) == len(inner_rows) and not np.any(
                valid[np.array(inner_rows)]
            )
            calibration["crossfit_full"] = (np.concatenate(inner_x), np.concatenate(inner_residual))
            local = frame.loc[valid].reset_index(drop=True)
            altered, selected = outage_frame(local, spec)
            for arm, (x, residual) in calibration.items():
                if time.monotonic() >= deadline:
                    raise BudgetStop("wall cap before copula fit")
                before = time.monotonic()
                calibration_rows = (
                    np.flatnonzero(train) if arm == "insample_full" else np.asarray(inner_rows)
                )
                calibration_path = OUT / f"calibration_{spec['id']}_{arm}.npz"
                np.savez_compressed(
                    calibration_path, row_indices=calibration_rows, physical_x=x, residual=residual
                )
                fitted = profile.fit_copula(x, residual)
                path = OUT / "03_model" / f"{spec['id']}_{arm}.npz"
                np.savez_compressed(path, **fitted)
                nx, ax = (
                    profile.physical_features(local, natural_c[valid]),
                    profile.physical_features(altered, outage_c[valid]),
                )
                natural[arm][valid] += profile.predict_copula(fitted, nx)
                altered_pred[arm][valid] += profile.predict_copula(fitted, ax)
                assert np.array_equal(
                    natural[arm][valid][~selected], altered_pred[arm][valid][~selected]
                )
                receipt = {
                    "fold": spec["id"],
                    "arm": arm,
                    "calibration_rows": len(residual),
                    "runtime_seconds": time.monotonic() - before,
                    "model_sha256": base.file_hash(path),
                    "calibration_sha256": base.file_hash(calibration_path),
                    "outer_labels_used": 0,
                    "outer_trained_label_models_used_in_crossfit": 0,
                    "inner_key_sha256": key_hash(frame.iloc[inner_rows])
                    if arm == "crossfit_full"
                    else None,
                }
                copula.append(receipt)
                np.savez_compressed(
                    OUT / f"replay_{spec['id']}_{arm}.npz",
                    natural_x=nx,
                    outage_x=ax,
                    natural_correction=natural[arm][valid] - natural_c[valid],
                    outage_correction=altered_pred[arm][valid] - outage_c[valid],
                )
            progress("candidate_fold_complete", fold=spec["id"])
        assert len(fits) == 72 and len(copula) == 16
        primary = folds == "B3"
        comparisons = {}
        for scope, selected in (
            ("primary_B3", primary),
            ("pooled", np.ones(len(frame), bool)),
            ("outage_interval", outaged),
        ):
            groups = cv.bootstrap_groups(frame.loc[selected], "P2", contract)
            predictions = altered_pred if scope == "outage_interval" else natural
            comparisons[scope] = {
                arm: cv.paired_bootstrap(
                    truth[selected],
                    predictions["C3"][selected],
                    predictions[arm][selected],
                    groups,
                    "rmse",
                    contract,
                )
                for arm in ("insample_full", "crossfit_full")
            }
            comparisons[scope]["crossfit_vs_insample"] = cv.paired_bootstrap(
                truth[selected],
                predictions["insample_full"][selected],
                predictions["crossfit_full"][selected],
                groups,
                "rmse",
                contract,
            )
        output = OUT / "evaluation.npz"
        np.savez_compressed(
            output,
            key=key_array(frame),
            truth=truth,
            fold=folds,
            layer=frame.layer.to_numpy(),
            time=frame.time.to_numpy(str),
            natural_T5_missing=~np.isfinite(frame.temp_5.to_numpy(float)),
            outage_mask=outaged,
            **{f"natural_{k}": v for k, v in natural.items()},
            **{f"outage_{k}": v for k, v in altered_pred.items()},
        )
        result = {
            "experiment_id": ID,
            "status": "COMPLETE_INTERNAL_8FOLD_CPU_ONLY",
            "training_pid": os.getpid(),
            "baseline_result_sha256": base.file_hash(REPORT / "baseline-result.json"),
            "support": seal["support"],
            "new_backbone_fits": len(fits),
            "new_copula_fits": len(copula),
            "new_full_fits": 0,
            "fit_receipts": fits,
            "copula_receipts": copula,
            "natural": panels(frame, truth, natural, folds, outaged, contract),
            "testmatched": panels(frame, truth, altered_pred, folds, outaged, contract),
            "comparisons": comparisons,
            "retained_mean_improvement_candidates": [
                arm
                for arm in ("insample_full", "crossfit_full")
                if comparisons["primary_B3"][arm]["candidate_retained"]
            ],
            "automatic_promotion": False,
            "runtime_seconds": time.monotonic() - start,
            "source_sha256_unchanged": base.file_hash(source) == cfg["source_sha256"],
            "evaluation_sha256": base.file_hash(output),
            "fit_receipts_sha256": base.file_hash(OUT / "fit-receipts.json"),
            "runner_sha256": seal["hashes"]["runner"],
            "config_sha256": seal["hashes"]["config"],
            "official_access_rows": 0,
            "csv_written": 0,
            "upload": 0,
            "evaluation_rows_deleted": 0,
            "gpu_used": False,
            "limitations": cfg["limitations"],
        }
        save(REPORT / "result.json", result)
        save(OUT / "terminal_result.json", result)
        progress("COMPLETE")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "backbone_fits": len(fits),
                    "copula_fits": len(copula),
                    "runtime_seconds": result["runtime_seconds"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        terminal = {
            "experiment_id": ID,
            "status": "TERMINAL_RESOURCE_BUDGET_LIMIT"
            if isinstance(exc, BudgetStop)
            else "TERMINAL_TECHNICAL_FAILURE",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_backbone_fits": len(fits),
            "completed_copula_fits": len(copula),
            "runtime_seconds": time.monotonic() - start,
            "automatic_restart": False,
            "scientific_decision": "NOT_ESTIMATED_INCOMPLETE",
            "official_access_rows": 0,
            "csv_written": 0,
            "upload": 0,
        }
        save(REPORT / "terminal-failure.json", terminal)
        save(OUT / "terminal_failure.json", terminal)
        print(json.dumps(terminal), flush=True)
        raise


def replay():
    cfg, contract, _ = settings()
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert fingerprints() == sealed["hashes"]
    source = install_guard()
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    assert os.getpid() != result["training_pid"]
    raw = np.load(OUT / "evaluation.npz", allow_pickle=False)
    assert base.file_hash(OUT / "evaluation.npz") == result["evaluation_sha256"]
    frame, truth = load_population(cfg)
    assert np.array_equal(key_array(frame), raw["key"]) and np.array_equal(truth, raw["truth"])
    checks, largest = [], 0.0
    for spec in contract["P2"]["folds"]:
        _, valid = masks(frame.time, spec)
        local = frame.loc[valid].reset_index(drop=True)
        altered, _ = outage_frame(local, spec)
        models = load_models(spec["id"], "outer", cfg["seeds"])
        for surface, query in (("natural", local), ("outage", altered)):
            c = cmean(models, query)
            assert np.array_equal(c, raw[f"{surface}_C3"][valid])
            for arm in ("insample_full", "crossfit_full"):
                path = OUT / "03_model" / f"{spec['id']}_{arm}.npz"
                fitted = dict(np.load(path, allow_pickle=False))
                prediction = c + profile.predict_copula(fitted, profile.physical_features(query, c))
                expected = raw[f"{surface}_{arm}"][valid]
                error = float(np.max(np.abs(prediction - expected)))
                largest = max(largest, error)
                checks.append(
                    {
                        "fold": spec["id"],
                        "surface": surface,
                        "arm": arm,
                        "n": len(query),
                        "exact": bool(np.array_equal(prediction, expected)),
                        "max_error": error,
                    }
                )
                assert np.array_equal(prediction, expected)
    assert base.file_hash(source) == cfg["source_sha256"]
    save(
        REPORT / "replay.json",
        {
            "status": "PASS",
            "pid": os.getpid(),
            "training_pid": result["training_pid"],
            "rows_per_surface": len(frame),
            "checks": checks,
            "max_error": largest,
            "scope": "fresh_process_released_context_outer_models_all_8fold_3arms_2surfaces_not_official",
            "new_fits": 0,
            "official_access_rows": 0,
            "csv_written": 0,
            "upload": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "REPLAY_PASS",
                "rows_per_surface": len(frame),
                "checks": len(checks),
                "maximum_error": largest,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("support", "seal", "execute", "replay"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "execute":
            execute()
            return
        if args.mode == "replay":
            replay()
            return
        cfg, contract, _ = settings()
        source = install_guard()
        frame, truth = load_population(cfg)
        support = support_audit(frame, truth, cfg, contract)
        assert base.file_hash(source) == cfg["source_sha256"]
        if args.mode == "seal":
            save(
                SEAL,
                {
                    "experiment_id": ID,
                    "hashes": fingerprints(),
                    "support": support,
                    "source_sha256": cfg["source_sha256"],
                    "sealed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                },
            )
            print("SEALED", flush=True)
        else:
            save(REPORT / "feature-support.json", support)
            print(json.dumps(support), flush=True)


if __name__ == "__main__":
    main()

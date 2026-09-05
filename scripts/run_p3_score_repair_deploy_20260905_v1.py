"""Rebuild source-only P3 training assets, then predict in a fresh process.

Stages: --prepare (train sources only), --train --gpu-approved (9 backbone fits),
--predict-replay (saved-model/local-case parity), and --predict-official
--official-approved (separately authorized anonymous inference, never uploads).
No existing submission or historical OOF is a runtime input.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostRegressor

from ocean_tabpfn3.offline import make_regressor
from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds
from p3_wave.data import LEADS, P3Data
from p3_wave.features import build_training_features, summarize_context
from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    ComponentLossRouter,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.persistence_shrink import LongLeadPersistenceShrink, apply_long_lead_persistence_shrink
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.validation import expand_leads

ROOT = Path(__file__).resolve().parents[1]
NAME = "p3_score_repair_deploy_20260905_v1"
OUT = ROOT / "artifacts" / NAME
REPORT = ROOT / "reports" / NAME
CONFIG = ROOT / "configs/experiments" / f"{NAME}.json"
KEYS = ["case_id", "station", "lead_h"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def stamp():
    return datetime.now(UTC).isoformat()


def boundary(source, allow_official=False):
    permitted = {source / "train_wave.csv", source / "train_atmos.csv"}
    if allow_official:
        permitted.add(source / "test_index.csv")

    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("Network disabled for offline reproduction")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        name = path.name.lower()
        if source in path.parents:
            mode = args[1]
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                raise PermissionError("source data are immutable")
        if "external_data" in path.parts or "hidden" in name:
            raise PermissionError("forbidden external or hidden path")
        if name.endswith(".csv") and path not in permitted:
            if not (allow_official and OUT in path.parents):
                raise PermissionError("CSV is not in the approved input/output allowlist")
        if name in {"test_features.parquet", "sample_submission.csv", "baseline_persistence.csv"}:
            raise PermissionError("existing official cache/sample/baseline forbidden")
        if name == "test_context.parquet" and not (allow_official and path == source / name):
            raise PermissionError("anonymous context not approved")
        if path.name == "oof.parquet" and OUT not in path.parents:
            raise PermissionError("historical OOF is not a runtime input")

    sys.addaudithook(guard)
    for variable in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TABPFN_NO_BROWSER",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        os.environ[variable] = "1"


def common(config, source):
    expected = config["source_files"]
    verified = {}
    for name, value in expected.items():
        verified[name] = sha(source / name)
        if verified[name] != value:
            raise ValueError(f"distributed source SHA mismatch: {name}")
    if sha(ROOT / config["reference_config"]) != config["reference_config_sha256"]:
        raise ValueError("reference recipe SHA mismatch")
    if sha(ROOT / config["reference_helpers"]) != config["reference_helpers_sha256"]:
        raise ValueError("reference helper SHA mismatch")
    return verified


def prepare(config, source):
    t0 = time.perf_counter()
    common(config, source)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "PREPARE_LOCK.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"started_utc": stamp(), "pid": os.getpid(), "config_sha256": sha(CONFIG)}, stream
        )
    wave = pd.read_csv(source / "train_wave.csv")
    atmos = pd.read_csv(source / "train_atmos.csv")
    for frame in (wave, atmos):
        frame["time"] = pd.to_datetime(frame.time, utc=True)
    empty = pd.DataFrame()
    data = P3Data(wave, atmos, empty, empty, empty, empty)

    def progress(done, total):
        value = {
            "stage": "source_feature_build",
            "completed_anchors": done,
            "total_anchors": total,
            "elapsed_seconds": time.perf_counter() - t0,
            "pid": os.getpid(),
            "gpu_used": False,
        }
        save(OUT / "progress.json", value)
        print(json.dumps(value), flush=True)

    built = build_training_features(data, dense_spacing_minutes=20, progress=progress)
    built.features.to_parquet(OUT / "train_features.parquet", index=False)
    built.anchors.to_parquet(OUT / "train_anchors.parquet", index=False)
    columns = compact_feature_columns(list(built.feature_columns))
    if len(columns) != 591 or len(built.anchors) != 24360:
        raise ValueError("source-only feature population differs")
    anchors = assign_storm_episodes_from_wave(built.anchors, wave)
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    _, selected, split = build_corrected_repeated_forward_folds(
        anchors, windows=recipe["validation"]["windows"], gap_hours=78, footprint_hours=72
    )
    selected.to_parquet(OUT / "validation_keys.parquet", index=False)
    replay = built.features.set_index("anchor_id").loc[selected.anchor_id].reset_index()
    replay.insert(0, "case_id", [f"LOCAL_{i:04d}" for i in range(len(replay))])
    replay.to_parquet(OUT / "replay_cases.parquet", index=False)
    save(OUT / "feature_columns.json", {"columns": columns})
    result = {
        "status": "PREPARED_FROM_DISTRIBUTED_TRAIN",
        "source_sha256": config["source_files"],
        "train_anchors": len(anchors),
        "feature_count": len(columns),
        "split_audit": split,
        "runtime_seconds": time.perf_counter() - t0,
        "runner_sha256": sha(Path(__file__)),
        "config_sha256": sha(CONFIG),
        "files": {
            name: sha(OUT / name)
            for name in [
                "train_features.parquet",
                "train_anchors.parquet",
                "validation_keys.parquet",
                "replay_cases.parquet",
                "feature_columns.json",
            ]
        },
        "historical_cache_reads": 0,
        "official_input_rows": 0,
        "fits": 0,
    }
    save(OUT / "prepare.json", result)
    save(REPORT / "prepare.json", result)
    print(json.dumps(result), flush=True)


def helpers(config):
    spec = importlib.util.spec_from_file_location(
        "p3_clean_fit_helpers", ROOT / config["reference_helpers"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows_for_cases(cases, columns):
    if cases.duplicated(["case_id", "station"]).any():
        raise ValueError("duplicate case")
    n = len(cases)
    matrix = cases.loc[:, columns].iloc[np.repeat(np.arange(n), len(LEADS))].reset_index(drop=True)
    matrix.insert(0, "station", np.repeat(cases.station.astype(str).to_numpy(), len(LEADS)))
    matrix.insert(1, "lead_h", np.tile(LEADS, n))
    current = np.repeat(cases.hs_current.to_numpy(dtype=float), len(LEADS))
    matrix.insert(2, "current_hs_for_residual", current)
    keys = pd.DataFrame(
        {
            "case_id": np.repeat(cases.case_id.to_numpy(), len(LEADS)),
            "station": np.repeat(cases.station.astype(str).to_numpy(), len(LEADS)),
            "lead_h": np.tile(LEADS, n),
        }
    )
    return matrix, current, keys


def tab_matrix(rows):
    selected = rows.loc[rows.lead_h.eq(6)].reset_index(drop=True)
    station = selected.station.map({"G-ORS": 0.0, "I-ORS": 1.0, "S-ORS": 2.0})
    if station.isna().any():
        raise ValueError("unknown station")
    numeric = selected.drop(columns=["station", "lead_h"]).to_numpy(dtype=np.float32)
    matrix = np.column_stack([station.to_numpy(dtype=np.float32), numeric])
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix.astype(np.float32, copy=False)


def predict_cases(cases, columns, single, multi, router, tab):
    rows, current, keys = rows_for_cases(cases, columns)
    cat_rows = rows.copy()
    cat_rows.station = cat_rows.station.astype(str)
    cat_rows.lead_h = cat_rows.lead_h.astype(str)
    single_values = np.clip(current + single.predict(cat_rows, thread_count=2), 0, 30)
    multi_matrix = cases[["station", *columns]].copy()
    multi_matrix.station = multi_matrix.station.astype(str)
    multi_values = np.clip(
        cases.hs_current.to_numpy()[:, None] + multi.predict(multi_matrix, thread_count=2), 0, 30
    )
    persistence = np.repeat(cases.hs_current.to_numpy()[:, None], len(LEADS), axis=1)
    components = np.stack([single_values.reshape(-1, 6), multi_values, persistence], axis=2)
    case_x = build_inference_router_features(
        cases.loc[:, OBSERVED_FEATURES],
        cases.station.to_numpy(str),
        cases.hs_current.to_numpy(),
        components,
    )
    metadata = pd.DataFrame(
        {
            "fold": "inference",
            "anchor_id": np.arange(len(cases)),
            "station": cases.station.to_numpy(str),
            "anchor_time": pd.NaT,
        }
    )
    row_x, row_meta, row_components = expand_case_router_features(case_x, metadata, components)
    weights = router.predict_weights(row_x)
    weights[~row_meta.lead_h.isin([12, 18, 24]).to_numpy()] = [0.5, 0.5, 0.0]
    routed = route_row_predictions(row_components, weights)
    baseline = apply_long_lead_persistence_shrink(
        routed,
        persistence.reshape(-1),
        row_meta.lead_h.to_numpy(),
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    six_prediction = np.clip(
        cases.hs_current.to_numpy() + np.asarray(tab.predict(tab_matrix(rows))).reshape(-1), 0, 30
    )
    candidate = baseline.copy()
    six = keys.lead_h.eq(6).to_numpy()
    candidate[six] = 0.75 * baseline[six] + 0.25 * six_prediction
    if not np.isfinite(np.column_stack([baseline, candidate])).all():
        raise ValueError("nonfinite inference")
    return keys, baseline, candidate


def train(config, source):
    import torch

    start = time.perf_counter()
    common(config, source)
    prepared = json.loads((OUT / "prepare.json").read_text(encoding="utf-8"))
    for name, value in prepared["files"].items():
        if sha(OUT / name) != value:
            raise ValueError("prepared feature artifact changed")
    if not torch.cuda.is_available():
        raise RuntimeError("approved GPU is unavailable")
    checkpoint = ROOT / config["tabpfn"]["checkpoint"]
    if sha(checkpoint) != config["tabpfn"]["sha256"]:
        raise ValueError("synthetic checkpoint SHA mismatch")
    receipt = json.loads((ROOT / config["tabpfn"]["license_receipt"]).read_text(encoding="utf-8"))
    if not all(
        receipt.get(k)
        for k in [
            "license_accepted_by_user",
            "competition_use_terms_reviewed",
            "synthetic_only_provenance_reviewed",
        ]
    ):
        raise PermissionError("synthetic pretrained exception evidence incomplete")
    with (OUT / "TRAIN_LOCK.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "started_utc": stamp(),
                "runner_sha256": sha(Path(__file__)),
                "config_sha256": sha(CONFIG),
            },
            stream,
        )
    base = helpers(config)
    recipe = json.loads((ROOT / config["reference_config"]).read_text(encoding="utf-8"))
    recipe["model"]["single"]["thread_count"] = 2
    recipe["model"]["multi"]["thread_count"] = 2
    features = pd.read_parquet(OUT / "train_features.parquet")
    anchors = pd.read_parquet(OUT / "train_anchors.parquet")
    columns = json.loads((OUT / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    wave = pd.read_csv(source / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave.time, utc=True)
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split = build_corrected_repeated_forward_folds(
        anchors, windows=recipe["validation"]["windows"], gap_hours=78, footprint_hours=72
    )
    receipts = []
    oof = []
    fit_count = 0
    try:
        for i, fold in enumerate(folds):
            progress = {
                "stage": "rebuild_router_oof",
                "fold": fold.name,
                "backbone_fits_completed": fit_count,
                "pid": os.getpid(),
                "elapsed_seconds": time.perf_counter() - start,
                "rss_gib": psutil.Process().memory_info().rss / 2**30,
            }
            save(OUT / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            frame, fit_receipt = base._fit_fold_components(
                fold=fold,
                fold_number=i,
                features=features,
                anchors=anchors,
                feature_columns=columns,
                config=recipe,
                model_dir=OUT / "models",
            )
            fit_count += 2
            receipts.append(fit_receipt)
            oof.append(frame)
            save(OUT / "fit-receipts.json", {"receipts": receipts, "backbone_fit_count": fit_count})
        rebuilt, detail, material = base._evaluate_fixed_structure(
            component_oof=pd.concat(oof, ignore_index=True),
            train_features=features,
            anchors=anchors,
            fold_order=tuple(f.name for f in folds),
            config=recipe,
            split_audit=split,
            expected_validation_ids=selected.anchor_id.to_numpy(),
        )
        rebuilt.to_parquet(OUT / "oof.parquet", index=False)
        reproduced_rmse = float(detail["metrics"]["final"]["rmse"])
        save(
            REPORT / "regenerated-oof.json",
            {
                "rmse_m": reproduced_rmse,
                "historical_reference_rmse_m": config["comparison"]["historical_clean_rmse_m"],
                "delta_numerical_and_runtime_retraining_m": reproduced_rmse
                - config["comparison"]["historical_clean_rmse_m"],
                "historical_oof_read": False,
                "metric_scope": "regenerated same historical 181 cases, not fresh confirmation",
                "oof_sha256": sha(OUT / "oof.parquet"),
                "detail": detail,
            },
        )
        save(
            OUT / "progress.json",
            {"stage": "full_catboost", "backbone_fits_completed": fit_count, "pid": os.getpid()},
        )
        full = OUT / "models/full"
        full.mkdir(parents=True, exist_ok=True)
        ids = anchors.anchor_id.to_numpy(dtype=np.int64)
        x, y, meta = expand_leads(features, anchors, ids, columns)
        t0 = time.perf_counter()
        single = base._single_model(recipe, recipe["model"]["full_train_seed"])
        single.fit(
            base._cat_frame(x),
            y,
            sample_weight=threshold_case_weights(meta.current_hs.to_numpy()),
            cat_features=[0, 1],
            verbose=False,
        )
        single.save_model(full / "single.cbm")
        fit_count += 1
        receipts.append(
            {"name": "full_single", "seconds": time.perf_counter() - t0, "rows": len(x)}
        )
        t0 = time.perf_counter()
        multi = base._multi_model(recipe, recipe["model"]["full_train_seed"])
        mx = features.set_index("anchor_id").loc[ids, ["station", *columns]].reset_index(drop=True)
        mx.station = mx.station.astype(str)
        multi.fit(
            mx,
            base._multi_target(anchors, ids),
            sample_weight=threshold_case_weights(
                anchors.set_index("anchor_id").loc[ids, "current_hs"].to_numpy()
            ),
            cat_features=[0],
            verbose=False,
        )
        multi.save_model(full / "multi.cbm")
        fit_count += 1
        receipts.append(
            {"name": "full_multi", "seconds": time.perf_counter() - t0, "rows": len(mx)}
        )
        router = ComponentLossRouter(base._router_config(recipe)).fit(
            material["row_features"], material["row_losses"]
        )
        joblib.dump(router, full / "router.joblib")
        save(OUT / "fit-receipts.json", {"receipts": receipts, "backbone_fit_count": fit_count})
        save(
            OUT / "progress.json",
            {"stage": "full_tabpfn_6h", "backbone_fits_completed": fit_count, "pid": os.getpid()},
        )
        bundled = full / "weights" / checkpoint.name
        bundled.parent.mkdir(exist_ok=True)
        shutil.copy2(checkpoint, bundled)
        save(full / "license-receipt.json", receipt)
        t0 = time.perf_counter()
        tab = make_regressor(
            bundled, seed=config["tabpfn"]["seed"], categorical_features_indices=[0], n_estimators=8
        )
        tx = tab_matrix(x)
        tab.fit(tx, y[x.lead_h.eq(6).to_numpy()])
        fit_count += 1
        tab_fit_seconds = time.perf_counter() - t0
        replay = pd.read_parquet(OUT / "replay_cases.parquet")
        _, expected_baseline, expected_candidate = predict_cases(
            replay, columns, single, multi, router, tab
        )
        np.savez_compressed(
            OUT / "replay_expected.npz", baseline=expected_baseline, candidate=expected_candidate
        )
        # Save fitted context/preprocessors separately from exact bundled synthetic weights.
        tab.model_path = bundled.relative_to(ROOT).as_posix()
        tab.save_fit_state(full / "tabpfn6.tabpfn_fit")
        receipts.append(
            {
                "name": "full_tabpfn6",
                "fit_seconds": tab_fit_seconds,
                "fit_plus_replay_save_seconds": time.perf_counter() - t0,
                "train_cases": len(tx),
            }
        )
        runtime = time.perf_counter() - start
        result = {
            "status": "TRAINED_AWAITING_FRESH_PROCESS_REPLAY",
            "experiment_id": NAME,
            "backbone_fit_count": fit_count,
            "catboost_fit_count": 8,
            "tabpfn_fit_count": 1,
            "router_fit_count": 3,
            "train_seconds": runtime,
            "prepare_seconds": prepared["runtime_seconds"],
            "elapsed_prepare_plus_train_seconds": runtime + prepared["runtime_seconds"],
            "under_6h_before_final_predict": runtime + prepared["runtime_seconds"] < 21600,
            "regenerated_baseline_oof_rmse_m": reproduced_rmse,
            "reference_drift_m": reproduced_rmse - config["comparison"]["historical_clean_rmse_m"],
            "official_input_rows": 0,
            "submission_rows": 0,
            "uploads": 0,
            "network_connections_allowed": 0,
            "historical_oof_or_csv_runtime_inputs": 0,
            "full_training_receipts": receipts,
            "files": {
                p.relative_to(OUT).as_posix(): sha(p) for p in full.rglob("*") if p.is_file()
            },
            "config_sha256": sha(CONFIG),
            "runner_sha256": sha(Path(__file__)),
        }
        save(OUT / "training-result.json", result)
        save(REPORT / "training-result.json", result)
        save(
            OUT / "progress.json",
            {
                "status": result["status"],
                "backbone_fit_count": fit_count,
                "elapsed_seconds": runtime,
            },
        )
        print(json.dumps(result), flush=True)
        del x, tx, mx, tab, single, multi, router
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        save(
            OUT / "terminal-failure.json",
            {
                "status": "TECHNICAL_FAILURE",
                "error_type": type(error).__name__,
                "error": str(error),
                "backbone_fits_completed": fit_count,
                "elapsed_seconds": time.perf_counter() - start,
                "retry_authorized": False,
            },
        )
        raise


def load_models():
    from tabpfn import TabPFNRegressor

    result = json.loads((OUT / "training-result.json").read_text(encoding="utf-8"))
    for relative, value in result["files"].items():
        if sha(OUT / relative) != value:
            raise ValueError("saved model or bundled weight SHA mismatch")
    full = OUT / "models/full"
    single = CatBoostRegressor().load_model(full / "single.cbm")
    multi = CatBoostRegressor().load_model(full / "multi.cbm")
    router = joblib.load(full / "router.joblib")
    tab = TabPFNRegressor.load_from_fit_state(full / "tabpfn6.tabpfn_fit", device="cuda")
    return single, multi, router, tab


def replay():
    t0 = time.perf_counter()
    models = load_models()
    cases = pd.read_parquet(OUT / "replay_cases.parquet")
    columns = json.loads((OUT / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    keys, baseline, candidate = predict_cases(cases, columns, *models)
    with np.load(OUT / "replay_expected.npz", allow_pickle=False) as expected:
        differences = {
            "baseline": float(np.max(np.abs(baseline - expected["baseline"]))),
            "candidate": float(np.max(np.abs(candidate - expected["candidate"]))),
        }
    if max(differences.values()) > 1e-6:
        raise ValueError("fresh-process inference differs from pre-save inference")
    result = {
        "status": "FRESH_PROCESS_REPLAY_PASS",
        "rows": len(keys),
        "cases": len(cases),
        "pid": os.getpid(),
        "maximum_absolute_difference_m": differences,
        "seconds": time.perf_counter() - t0,
        "official_input_rows": 0,
        "backbone_fits": 0,
        "same_case_only": True,
        "network_connections_allowed": 0,
    }
    save(OUT / "replay-qa.json", result)
    save(REPORT / "replay-qa.json", result)
    print(json.dumps(result), flush=True)


def predict_official(source):
    if not (OUT / "replay-qa.json").exists():
        raise PermissionError("fresh-process replay must pass first")
    training = json.loads((OUT / "training-result.json").read_text(encoding="utf-8"))
    replay_qa = json.loads((OUT / "replay-qa.json").read_text(encoding="utf-8"))
    if replay_qa["status"] != "FRESH_PROCESS_REPLAY_PASS":
        raise PermissionError("fresh-process replay has not passed")
    prior_seconds = training["elapsed_prepare_plus_train_seconds"] + replay_qa["seconds"]
    if prior_seconds >= 21600:
        raise RuntimeError("six-hour preparation budget exhausted before official inference")
    started = time.perf_counter()
    with (OUT / "PREDICT_LOCK.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"approved_official_inference": True, "started_utc": stamp(), "pid": os.getpid()},
            stream,
        )
    context = pd.read_parquet(source / "test_context.parquet")
    index = pd.read_csv(source / "test_index.csv")
    if list(index) != KEYS or len(index) != 1200 or index.duplicated(KEYS).any():
        raise ValueError("official index schema changed")
    rows = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        if (
            len(group) != 289
            or group.station.nunique() != 1
            or not np.array_equal(group.step_minute.to_numpy(), np.arange(-2880, 1, 10))
        ):
            raise ValueError("case-local context contract mismatch")
        rows.append(
            {"case_id": case_id, "station": str(group.station.iloc[0]), **summarize_context(group)}
        )
    cases = pd.DataFrame(rows)
    order = index[["case_id", "station"]].drop_duplicates()
    cases = order.merge(cases, on=["case_id", "station"], validate="one_to_one")
    if len(cases) != 200:
        raise ValueError("official case count differs")
    columns = json.loads((OUT / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    keys, baseline, candidate = predict_cases(cases, columns, *load_models())
    outputs = {}
    for label, values in [("clean_baseline", baseline), ("tabpfn25_6h_only", candidate)]:
        frame = keys.copy()
        frame["hs_pred"] = values
        frame = index.merge(frame, on=KEYS, validate="one_to_one", how="left")
        if (
            len(frame) != 1200
            or not np.isfinite(frame.hs_pred).all()
            or not frame.hs_pred.between(0, 30).all()
        ):
            raise ValueError("prediction schema/range failure")
        path = OUT / "candidates" / f"{label}.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
        reread = pd.read_csv(path)
        if not reread[KEYS].equals(index[KEYS]) or not np.allclose(
            reread.hs_pred, frame.hs_pred, rtol=0, atol=1e-12
        ):
            raise ValueError("roundtrip key/order/value mismatch")
        outputs[label] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "rows": 1200,
            "key_order_exact": True,
            "finite_range_0_30": True,
        }
    seconds = time.perf_counter() - started
    result = {
        "status": "TWO_LOCAL_CANDIDATES_PREPARED_NOT_UPLOADED",
        "outputs": outputs,
        "official_context_rows": len(context),
        "official_index_rows": len(index),
        "hidden_rows": 0,
        "sample_rows": 0,
        "uploads": 0,
        "prediction_seconds": seconds,
        "fresh_process_replay_seconds": replay_qa["seconds"],
        "source_prepare_train_replay_predict_seconds": prior_seconds + seconds,
        "under_six_hours": prior_seconds + seconds < 21600,
        "timing_hardware_scope": "current RTX 5090 host; organizer hardware not confirmed",
        "historical_oof_or_csv_runtime_inputs": 0,
    }
    save(OUT / "prediction-result.json", result)
    save(REPORT / "prediction-result.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    for name in ["prepare", "train", "predict-replay", "predict-official"]:
        stage.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--gpu-approved", action="store_true")
    parser.add_argument("--official-approved", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (args.train or args.predict_replay or args.predict_official) and not args.gpu_approved:
        raise PermissionError("root GPU allocation required")
    if args.predict_official and not args.official_approved:
        raise PermissionError("separate official-input approval required")
    boundary(source, allow_official=args.predict_official and args.official_approved)
    if args.prepare:
        prepare(config, source)
    elif args.train:
        train(config, source)
    elif args.predict_replay:
        replay()
    else:
        predict_official(source)


if __name__ == "__main__":
    main()

"""Standalone source-to-model-to-answer P3 clean package.

Run from any cwd with Python -I. The only external data path is P3_DATA_DIR.
No repository import, old model, old OOF, old answer, or network is required.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUT = ROOT
DATA = ROOT / "01_data"
MODELS = ROOT / "03_model"
WORK = ROOT / "04_logs" / "validation"
ANSWER = ROOT / "05_answer"
REPORT = ROOT / "04_logs" / "receipts"
CONFIG = CODE / "config.json"
sys.path.insert(0, str(CODE))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402
from p3_clean import backbone as base  # noqa: E402
from p3_clean.corrected_repeated_forward import build_corrected_repeated_forward_folds  # noqa: E402
from p3_clean.data import LEADS, P3Data  # noqa: E402
from p3_clean.episodes import assign_storm_episodes_from_wave  # noqa: E402
from p3_clean.features import (  # noqa: E402
    BASE_COLUMNS,
    DIRECTION_COLUMNS,
    build_training_features,
    summarize_context,
)
from p3_clean.loss_router import (  # noqa: E402
    OBSERVED_FEATURES,
    ComponentLossRouter,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from p3_clean.models import compact_feature_columns, threshold_case_weights  # noqa: E402
from p3_clean.persistence_shrink import (  # noqa: E402
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_clean.validation import expand_leads  # noqa: E402

KEYS = ["case_id", "station", "lead_h"]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value, *, progress=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if progress else "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)


def stamp():
    return datetime.now(UTC).isoformat()


def environment():
    return {
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "scipy",
                "scikit-learn",
                "catboost",
                "joblib",
            )
        },
    }


def guard(source, official=False):
    source_allowed = {source / name for name in ("train_wave.csv", "train_atmos.csv")}
    if official:
        source_allowed |= {source / "test_context.parquet", source / "test_index.csv"}

    def hook(event, args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise PermissionError(
                "network forbidden by Python audit hook; not OS sandbox certification"
            )
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        owned = ROOT in path.parents
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if source in path.parents and (path not in source_allowed or writing):
            raise PermissionError("source outside immutable stage-specific allowlist")
        if any(
            x in path.name.lower()
            for x in ("hidden", "credentials", "refined_alpha", "axis_contract")
        ):
            raise PermissionError("forbidden hidden or coefficient ancestry")
        if (
            path.suffix.lower()
            in {".parquet", ".npz", ".cbm", ".joblib", ".ckpt", ".pt", ".tabpfn_fit", ".csv"}
            and not owned
            and path not in source_allowed
        ):
            raise PermissionError("non-package data/model/OOF/answer input forbidden")
        if path.suffix.lower() == ".csv" and owned and path.parent != ANSWER:
            raise PermissionError("generated CSV outside answer folder")
        if path.parent == ANSWER and not official:
            raise PermissionError("answer access before approved inference")

    sys.addaudithook(hook)


def verify(config, source, *, prepared=False):
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = {}
    for name, expected in manifest["sha256"].items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or sha(path) != expected:
            raise ValueError("portable package code/recipe manifest mismatch")
        checked[name] = expected
    local = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("p3_wave"):
            raise ValueError("original repository import forbidden")
        if name.startswith("p3_clean") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if CODE not in path.parents:
                raise ValueError("package dependency escaped 02_code")
            local[name] = path.relative_to(ROOT).as_posix()
    if not local:
        raise ValueError("local dependency closure empty")
    if environment() != config["environment"]:
        raise ValueError("environment differs from fixed tested versions; do not silently change")
    for name, expected in config["source_files"].items():
        if sha(source / name) != expected:
            raise ValueError("distributed source SHA mismatch")
    if prepared:
        seal = json.loads((OUT / "seal.json").read_text(encoding="utf-8"))
        if sha(CONFIG) != seal["config_sha256"] or sha(Path(__file__)) != seal["runner_sha256"]:
            raise ValueError("sealed package runner/config changed")
    return {"package_sha256": checked, "loaded_local_modules": local, "environment": environment()}


def canonical_timeline(selected, windows):
    """Audit targets available at canonical router fit time (fold start).

    Context-start separation is reported but is not a new blocking rule.
    """
    timeline, completed = [], []
    for name, start, _ in windows:
        current = selected.loc[selected.fold.eq(name)]
        if completed:
            past = selected.loc[selected.fold.isin(completed)]
            target_ready = past.anchor_time.max() + pd.Timedelta(hours=24)
            fold_start = pd.Timestamp(start, tz="UTC")
            gap = (fold_start - target_ready).total_seconds() / 3600
            if gap <= 0:
                raise ValueError("canonical prior OOF targets unavailable before fold start")
            context_gap = (
                current.anchor_time.min() - pd.Timedelta(hours=48) - target_ready
            ).total_seconds() / 3600
            timeline.append(
                {
                    "fold": name,
                    "past_cases": len(past),
                    "target_ready_fold_start_gap_hours": gap,
                    "target_ready_context_gap_hours_diagnostic_only": context_gap,
                }
            )
        completed.append(name)
    return timeline


def prepare(config, source):
    started = time.perf_counter()
    if (OUT / "PREPARE_LOCK.json").exists() or (WORK.exists() and any(WORK.iterdir())):
        raise RuntimeError("prepare attempt consumed; preserve outputs and do not restart")
    verified = verify(config, source)
    for folder in (DATA, MODELS, WORK, ANSWER, REPORT):
        folder.mkdir(parents=True, exist_ok=True)
    if any(MODELS.iterdir()) or any(ANSWER.iterdir()):
        raise ValueError("03_model and 05_answer must begin empty")
    save(
        OUT / "PREPARE_LOCK.json", {"pid": os.getpid(), "created_utc": stamp(), "empty_model": True}
    )
    save(
        DATA / "source-reference.json",
        {
            "path_contract": "P3_DATA_DIR supplied by operator; not embedded in package",
            "files_sha256": config["source_files"],
            "raw_data_copied": False,
        },
    )
    save(
        OUT / "seal.json",
        {
            "pid": os.getpid(),
            "created_utc": stamp(),
            "runner_sha256": sha(Path(__file__)),
            "config_sha256": sha(CONFIG),
            "verified": verified,
            "empty_03_model": True,
            "old_model_cache_oof_answer_reads": 0,
            "official_rows": 0,
        },
    )
    wave, atmos = (pd.read_csv(source / name) for name in ("train_wave.csv", "train_atmos.csv"))
    for frame in (wave, atmos):
        frame.time = pd.to_datetime(frame.time, utc=True)
    empty = pd.DataFrame()
    data = P3Data(wave, atmos, empty, empty, empty, empty)

    def progress(done, total):
        receipt = {
            "stage": "SOURCE_FEATURE_BUILD",
            "pid": os.getpid(),
            "completed_anchors": done,
            "total_anchors": total,
            "elapsed_seconds": time.perf_counter() - started,
            "gpu": False,
        }
        save(OUT / "progress.json", receipt, progress=True)
        print(json.dumps(receipt), flush=True)
        if receipt["elapsed_seconds"] > 21600:
            raise TimeoutError("six-hour execution cap")

    built = build_training_features(data, dense_spacing_minutes=20, progress=progress)
    columns = compact_feature_columns(list(built.feature_columns))
    if len(columns) != 591 or len(built.anchors) != 24360:
        raise ValueError("source population or feature count differs")
    built.features.to_parquet(WORK / "train_features.parquet", index=False)
    built.anchors.to_parquet(WORK / "train_anchors.parquet", index=False)
    recipe = json.loads((CODE / "recipe.json").read_text(encoding="utf-8"))
    anchors = assign_storm_episodes_from_wave(built.anchors, wave)
    _, selected, split = build_corrected_repeated_forward_folds(
        anchors, windows=recipe["validation"]["windows"], gap_hours=78, footprint_hours=72
    )
    if len(selected) != 181:
        raise ValueError("fixed historical selection differs")
    selected.to_parquet(WORK / "validation_keys.parquet", index=False)
    replay_cases = built.features.set_index("anchor_id").loc[selected.anchor_id].reset_index()
    replay_cases.insert(0, "case_id", [f"LOCAL_{i:04d}" for i in range(len(replay_cases))])
    replay_cases.to_parquet(WORK / "replay_cases.parquet", index=False)
    save(WORK / "feature_columns.json", {"columns": columns})
    timeline = canonical_timeline(selected, recipe["validation"]["windows"])
    receipt = {
        "status": "PREPARED_SOURCE_ONLY",
        "pid": os.getpid(),
        "seconds": time.perf_counter() - started,
        "train_anchors": 24360,
        "features": 591,
        "cases": 181,
        "fits": 0,
        "old_cache_reads": 0,
        "official_rows": 0,
        "empty_03_model_after_prepare": not any(MODELS.iterdir()),
        "split_audit": split,
        "prior_oof_actual_time_audit": timeline,
        "files": {p.name: sha(p) for p in WORK.iterdir() if p.is_file()},
        "source_sha256": config["source_files"],
        "seal_sha256": sha(OUT / "seal.json"),
    }
    save(REPORT / "prepare.json", receipt)
    print(json.dumps({"status": receipt["status"], "seconds": receipt["seconds"]}), flush=True)


def rows_for_cases(cases, columns):
    if cases.duplicated(["case_id", "station"]).any():
        raise ValueError("duplicate cases")
    matrix = cases[columns].iloc[np.repeat(np.arange(len(cases)), 6)].reset_index(drop=True)
    matrix.insert(0, "station", np.repeat(cases.station.astype(str).to_numpy(), 6))
    matrix.insert(1, "lead_h", np.tile(LEADS, len(cases)))
    current = np.repeat(cases.hs_current.to_numpy(dtype=float), 6)
    matrix.insert(2, "current_hs_for_residual", current)
    keys = pd.DataFrame(
        {
            "case_id": np.repeat(cases.case_id.to_numpy(), 6),
            "station": matrix.station,
            "lead_h": matrix.lead_h,
        }
    )
    return matrix, current, keys


def predict_cases(cases, columns, single, multi, router):
    matrix, current, keys = rows_for_cases(cases, columns)
    matrix.station, matrix.lead_h = matrix.station.astype(str), matrix.lead_h.astype(str)
    single_prediction = np.clip(current + single.predict(matrix, thread_count=2), 0, 30)
    multi_matrix = cases[["station", *columns]].copy()
    multi_matrix.station = multi_matrix.station.astype(str)
    multi_prediction = np.clip(
        cases.hs_current.to_numpy()[:, None] + multi.predict(multi_matrix, thread_count=2), 0, 30
    )
    persistence = np.repeat(cases.hs_current.to_numpy()[:, None], 6, axis=1)
    components = np.stack([single_prediction.reshape(-1, 6), multi_prediction, persistence], axis=2)
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
    weights = np.asarray(router.predict_weights(row_x), dtype=float)
    weights[~row_meta.lead_h.isin([12, 18, 24]).to_numpy()] = [0.5, 0.5, 0]
    routed = route_row_predictions(row_components, weights)
    prediction = apply_long_lead_persistence_shrink(
        routed,
        persistence.reshape(-1),
        row_meta.lead_h.to_numpy(),
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    if not np.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 30).any():
        raise ValueError("prediction range/finite failure")
    return keys, prediction


def train(config, source):
    verify(config, source, prepared=True)
    if any(MODELS.iterdir()) or (OUT / "TRAIN_LOCK.json").exists():
        raise RuntimeError("03_model must be empty for exactly-once training")
    prepared = json.loads((REPORT / "prepare.json").read_text(encoding="utf-8"))
    for name, expected in prepared["files"].items():
        if sha(WORK / name) != expected:
            raise ValueError("new source-generated feature changed")
    started = time.perf_counter()
    save(
        OUT / "TRAIN_LOCK.json",
        {
            "pid": os.getpid(),
            "created_utc": stamp(),
            "empty_03_model": True,
            "gpu_authorized": True,
            "max_backbone_fits": 8,
            "max_router_fits": 3,
        },
    )
    recipe = json.loads((CODE / "recipe.json").read_text(encoding="utf-8"))
    for kind in ("single", "multi"):
        recipe["model"][kind]["thread_count"] = 2
    if recipe["router"]["alpha"] != 10 or recipe["shrink"] != {
        "active_leads": [12, 18, 24],
        "persistence_weight": 0.2,
    }:
        raise ValueError("fixed clean regularization/shrink contract changed")
    features = pd.read_parquet(WORK / "train_features.parquet")
    anchors = pd.read_parquet(WORK / "train_anchors.parquet")
    columns = json.loads((WORK / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    anchors = assign_storm_episodes_from_wave(anchors, pd.read_csv(source / "train_wave.csv"))
    folds, selected, split = build_corrected_repeated_forward_folds(
        anchors, windows=recipe["validation"]["windows"], gap_hours=78, footprint_hours=72
    )
    fits, blocks = [], []

    def progress(stage, complete):
        receipt = {
            "stage": stage,
            "pid": os.getpid(),
            "completed_backbone_fits": complete,
            "max_backbone_fits": 8,
            "elapsed_train_seconds": time.perf_counter() - started,
            "gpu_owner": "P3",
            "cpu_threads": 2,
        }
        save(OUT / "progress.json", receipt, progress=True)
        print(json.dumps(receipt), flush=True)
        if prepared["seconds"] + receipt["elapsed_train_seconds"] > 21600:
            raise TimeoutError("six-hour source/train budget exhausted; no restart")

    for index, fold in enumerate(folds):
        progress(f"HISTORICAL_{fold.name}", 2 * index)
        frame, receipt = base._fit_fold_components(
            fold=fold,
            fold_number=index,
            features=features,
            anchors=anchors,
            feature_columns=columns,
            config=recipe,
            model_dir=MODELS / "historical",
        )
        blocks.append(frame)
        fits.append(receipt)
        save(
            OUT / "fit-receipts.json",
            {"historical": fits, "completed_backbone_fits": len(fits) * 2},
            progress=True,
        )
        gc.collect()
    rebuilt, detail, material = base._evaluate_fixed_structure(
        component_oof=pd.concat(blocks, ignore_index=True),
        train_features=features,
        anchors=anchors,
        fold_order=tuple(f.name for f in folds),
        config=recipe,
        split_audit=split,
        expected_validation_ids=selected.anchor_id.to_numpy(),
    )
    rebuilt.to_parquet(WORK / "oof.parquet", index=False)
    oof_rmse = float(np.sqrt(np.square(rebuilt.target_hs - rebuilt.final_prediction).mean()))
    if not np.isclose(oof_rmse, detail["metrics"]["final"]["rmse"], rtol=0, atol=1e-12):
        raise ValueError("independent OOF RMSE arithmetic mismatch")
    save(
        REPORT / "regenerated-oof.json",
        {
            "status": "REGENERATED_SOURCE_ONLY",
            "rmse_m": oof_rmse,
            "prior_scalar_rmse_m_qa_only": config["expected_clean_rmse_m_qa_only"],
            "rmse_drift_m": oof_rmse - config["expected_clean_rmse_m_qa_only"],
            "rows": len(rebuilt),
            "cases": rebuilt.anchor_id.nunique(),
            "sse_m2": float(np.square(rebuilt.target_hs - rebuilt.final_prediction).sum()),
            "old_oof_read": False,
            "oof_sha256": sha(WORK / "oof.parquet"),
            "detail": detail,
        },
    )
    progress("FULL_SINGLE", 6)
    full = MODELS / "full"
    full.mkdir(parents=True)
    ids = anchors.anchor_id.to_numpy(dtype=np.int64)
    x, y, meta = expand_leads(features, anchors, ids, columns)
    begin = time.perf_counter()
    single = base._single_model(recipe, recipe["model"]["full_train_seed"])
    single.fit(
        base._cat_frame(x),
        y,
        sample_weight=threshold_case_weights(meta.current_hs.to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single.save_model(full / "single.cbm")
    single_seconds = time.perf_counter() - begin
    progress("FULL_MULTI", 7)
    begin = time.perf_counter()
    mx = features.set_index("anchor_id").loc[ids, ["station", *columns]].reset_index(drop=True)
    mx.station = mx.station.astype(str)
    multi = base._multi_model(recipe, recipe["model"]["full_train_seed"])
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
    multi_seconds = time.perf_counter() - begin
    progress("FULL_ROUTER_AND_REPLAY_EXPECTATION", 8)
    router = ComponentLossRouter(base._router_config(recipe)).fit(
        material["row_features"], material["row_losses"]
    )
    joblib.dump(router, full / "router.joblib")
    cases = pd.read_parquet(WORK / "replay_cases.parquet")
    _, expectation = predict_cases(cases, columns, single, multi, router)
    np.savez_compressed(WORK / "replay_expected.npz", prediction=expectation)
    model_hashes = {
        path.relative_to(OUT).as_posix(): sha(path) for path in MODELS.rglob("*") if path.is_file()
    }
    elapsed = time.perf_counter() - started
    result = {
        "status": "TRAINING_COMPLETE_FROM_EMPTY_MODEL",
        "pid": os.getpid(),
        "empty_03_model_verified": True,
        "backbone_fits": 8,
        "historical_backbone_fits": 6,
        "full_backbone_fits": 2,
        "prequential_router_fits": 2,
        "full_router_fits": 1,
        "full_single_rows": len(x),
        "full_multi_cases": len(mx),
        "full_single_seconds": single_seconds,
        "full_multi_seconds": multi_seconds,
        "train_seconds": elapsed,
        "prepare_seconds": prepared["seconds"],
        "prepare_train_seconds": elapsed + prepared["seconds"],
        "regenerated_oof_rmse_m": oof_rmse,
        "model_sha256": model_hashes,
        "files_sha256": {
            "04_logs/validation/replay_cases.parquet": sha(WORK / "replay_cases.parquet"),
            "04_logs/validation/replay_expected.npz": sha(WORK / "replay_expected.npz"),
            "04_logs/validation/feature_columns.json": sha(WORK / "feature_columns.json"),
        },
        "runner_sha256": sha(Path(__file__)),
        "config_sha256": sha(CONFIG),
        "source_sha256": config["source_files"],
        "old_model_cache_oof_answer_reads": 0,
        "pretrained_weights": 0,
        "official_rows": 0,
        "csv_rows": 0,
        "uploads": 0,
        "six_hour_checked_after_each_historical_pair_and_full_model": True,
        "organizer_hardware_verified": False,
    }
    if result["prepare_train_seconds"] > 21600:
        raise TimeoutError("six-hour total limit exceeded")
    save(REPORT / "training-result.json", result)
    save(
        OUT / "TRAIN_TERMINAL.json",
        {
            "status": "COMPLETE",
            "result_sha256": sha(REPORT / "training-result.json"),
            "backbone_fits": 8,
            "router_fits": 3,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seconds": result["prepare_train_seconds"],
                "backbone_fits": 8,
                "router_fits": 3,
            }
        ),
        flush=True,
    )


def load_models(result):
    for name, expected in {**result["model_sha256"], **result["files_sha256"]}.items():
        if sha(OUT / name) != expected:
            raise ValueError("saved model or replay input changed")
    full = MODELS / "full"
    return (
        CatBoostRegressor().load_model(full / "single.cbm"),
        CatBoostRegressor().load_model(full / "multi.cbm"),
        joblib.load(full / "router.joblib"),
    )


def replay(config, source):
    started = time.perf_counter()
    verify(config, source, prepared=True)
    training = json.loads((REPORT / "training-result.json").read_text(encoding="utf-8"))
    if training["pid"] == os.getpid():
        raise ValueError("replay must be a fresh process")
    cases = pd.read_parquet(WORK / "replay_cases.parquet")
    columns = json.loads((WORK / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    keys, prediction = predict_cases(cases, columns, *load_models(training))
    with np.load(WORK / "replay_expected.npz", allow_pickle=False) as expected:
        error = float(np.max(np.abs(prediction - expected["prediction"])))
    if error != 0:
        raise ValueError("fresh process clean inference does not exactly replay")
    receipt = {
        "status": "PASS",
        "fresh_pid": os.getpid(),
        "training_pid": training["pid"],
        "cases": len(cases),
        "rows": len(keys),
        "max_abs_prediction_error_m": error,
        "seconds": time.perf_counter() - started,
        "training_result_sha256": sha(REPORT / "training-result.json"),
        "official_rows": 0,
        "same_environment_fresh_process_only": True,
    }
    save(REPORT / "fresh-process-replay.json", receipt)
    print(json.dumps(receipt), flush=True)


def official_frame(source, columns, models):
    allowed = ["case_id", "station", "step_minute", *BASE_COLUMNS, *DIRECTION_COLUMNS]
    context = pd.read_parquet(source / "test_context.parquet", columns=allowed)
    index = pd.read_csv(source / "test_index.csv", usecols=KEYS)
    if (
        len(index) != 1200
        or index.duplicated(KEYS).any()
        or not index.groupby(["case_id", "station"])
        .lead_h.agg(lambda x: sorted(x) == list(LEADS))
        .all()
    ):
        raise ValueError("official public keys invalid")
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        if (
            len(group) != 289
            or group.station.nunique() != 1
            or not np.array_equal(group.step_minute, np.arange(-2880, 1, 10))
        ):
            raise ValueError("anonymous case context invalid")
        records.append(
            {"case_id": case_id, "station": str(group.station.iloc[0]), **summarize_context(group)}
        )
    cases = pd.DataFrame(records)
    cases = (
        index[["case_id", "station"]]
        .drop_duplicates()
        .merge(cases, on=["case_id", "station"], how="outer", validate="one_to_one", indicator=True)
    )
    if len(cases) != 200 or not cases._merge.eq("both").all():
        raise ValueError("official case population mismatch")
    cases = cases.drop(columns="_merge")
    keys, prediction = predict_cases(cases, columns, *models)
    keys["hs_pred"] = prediction
    frame = index.merge(keys, on=KEYS, how="left", validate="one_to_one")
    if (
        not frame[KEYS].equals(index[KEYS])
        or len(frame) != 1200
        or not np.isfinite(frame.hs_pred).all()
        or not frame.hs_pred.between(0, 30).all()
    ):
        raise ValueError("answer schema/order/finite/range invalid")
    return frame, len(context)


def csv_bytes(frame):
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False)
    return stream.getvalue().encode("utf-8")


def inference(config, source, *, verify_answer=False):
    started = time.perf_counter()
    verify(config, source, prepared=True)
    training = json.loads((REPORT / "training-result.json").read_text(encoding="utf-8"))
    replay_receipt = json.loads((REPORT / "fresh-process-replay.json").read_text(encoding="utf-8"))
    if replay_receipt["status"] != "PASS" or replay_receipt["training_result_sha256"] != sha(
        REPORT / "training-result.json"
    ):
        raise ValueError("fresh-process replay must pass before official input access")
    if training["pid"] == os.getpid():
        raise ValueError("inference must not share training process")
    independent = json.loads((REPORT / "training-independent-qa.json").read_text(encoding="utf-8"))
    if independent["status"] != "PASS" or independent["training_result_sha256"] != sha(
        REPORT / "training-result.json"
    ):
        raise ValueError("independent training QA required before official access")
    if not verify_answer:
        if any(ANSWER.iterdir()):
            raise RuntimeError("05_answer must start empty")
        save(
            OUT / "INFERENCE_LOCK.json",
            {"pid": os.getpid(), "created_utc": stamp(), "local_only_authorized": True},
        )
    columns = json.loads((WORK / "feature_columns.json").read_text(encoding="utf-8"))["columns"]
    frame, context_rows = official_frame(source, columns, load_models(training))
    payload = csv_bytes(frame)
    digest = hashlib.sha256(payload).hexdigest()
    output = ANSWER / "submission.csv"
    total = (
        training["prepare_train_seconds"]
        + replay_receipt["seconds"]
        + time.perf_counter()
        - started
    )
    began = json.loads((OUT / "PREPARE_LOCK.json").read_text(encoding="utf-8"))["created_utc"]
    elapsed_wall = (datetime.now(UTC) - datetime.fromisoformat(began)).total_seconds()
    if total > 21600 or elapsed_wall > 21600:
        raise TimeoutError("six-hour total generation cap exceeded")
    if verify_answer:
        prior = json.loads((REPORT / "answer-qa.json").read_text(encoding="utf-8"))
        if prior["pid"] == os.getpid() or digest != sha(output) or digest != prior["sha256"]:
            raise ValueError("separate-process answer bytes replay differs")
        receipt = {
            "status": "PASS",
            "pid": os.getpid(),
            "initial_inference_pid": prior["pid"],
            "rows": len(frame),
            "sha256": digest,
            "exact_csv_bytes": True,
            "seconds": time.perf_counter() - started,
            "answer_qa_sha256": sha(REPORT / "answer-qa.json"),
            "official_context_rows": context_rows,
            "official_index_rows": len(frame),
            "sample_rows": 0,
            "hidden_rows": 0,
            "uploads": 0,
        }
        save(REPORT / "answer-replay-qa.json", receipt)
    else:
        with output.open("xb") as stream:
            stream.write(payload)
        loaded = pd.read_csv(output)
        if not loaded[KEYS].equals(frame[KEYS]) or not np.allclose(
            loaded.hs_pred, frame.hs_pred, rtol=0, atol=1e-12
        ):
            raise ValueError("CSV roundtrip differs")
        receipt = {
            "status": "LOCAL_ANSWER_GENERATED_NOT_UPLOADED",
            "pid": os.getpid(),
            "training_pid": training["pid"],
            "rows": len(frame),
            "cases": 200,
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": sha(output),
            "prior_clean_sha256_qa_only": config["expected_clean_csv_sha256_qa_only"],
            "exact_prior_clean_sha_match": sha(output)
            == config["expected_clean_csv_sha256_qa_only"],
            "schema": [*KEYS, "hs_pred"],
            "order_keys_exact": True,
            "duplicate_keys": 0,
            "finite_0_30": True,
            "official_context_rows": context_rows,
            "official_index_rows": 1200,
            "sample_rows": 0,
            "hidden_rows": 0,
            "old_answer_rows_read": 0,
            "uploads": 0,
            "seconds": time.perf_counter() - started,
            "source_train_replay_inference_seconds": total,
            "under_six_hours_current_machine": total < 21600,
            "training_result_sha256": sha(REPORT / "training-result.json"),
            "fresh_replay_sha256": sha(REPORT / "fresh-process-replay.json"),
            "empty_model_to_answer_regeneration_completed": True,
            "final_organizer_environment_verification": False,
        }
        save(REPORT / "answer-qa.json", receipt)
    save(
        REPORT / ("answer-replay-wall.json" if verify_answer else "source-to-answer-wall.json"),
        {
            "elapsed_wall_seconds_including_process_gaps": elapsed_wall,
            "under_six_hours": elapsed_wall < 21600,
            "package_only": True,
        },
    )
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--RUN_TRAINING", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--RUN_INFERENCE", action="store_true")
    mode.add_argument("--verify-answer", action="store_true")
    parser.add_argument("--gpu-approved", action="store_true")
    parser.add_argument("--official-approved", action="store_true")
    args = parser.parse_args()
    if args.RUN_TRAINING and not args.gpu_approved:
        raise PermissionError("allocate exclusive GPU before training")
    official = args.RUN_INFERENCE or args.verify_answer
    if official and not args.official_approved:
        raise PermissionError("local reproduction answer authorization required")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    if source == ROOT or ROOT in source.parents or source in ROOT.parents:
        raise ValueError("source dataset and package must be disjoint")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    guard(source, official)
    try:
        if args.prepare:
            prepare(config, source)
        elif args.RUN_TRAINING:
            if not (REPORT / "prepare.json").exists():
                prepare(config, source)
            train(config, source)
        elif args.replay:
            replay(config, source)
        else:
            inference(config, source, verify_answer=args.verify_answer)
    except Exception as exc:
        failure = OUT / "FAILURE_RECEIPT.json"
        if not failure.exists():
            save(
                failure,
                {
                    "status": "TECHNICAL_FAILURE_NO_RESTART",
                    "pid": os.getpid(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "created_utc": stamp(),
                },
            )
        raise


if __name__ == "__main__":
    main()

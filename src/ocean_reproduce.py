"""Fail-closed orchestration for the three frozen competition submissions."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_SHA = {
    "P1": "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    "P2": "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    "P3": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
}

P1_MODEL = Path("artifacts/runs/20260813T155254+0900_train_378a4e89/model.joblib")
P1_CONFIG = Path("artifacts/runs/20260813T155254+0900_train_378a4e89/config.toml")
P1_SELECTION = Path("artifacts/runs/20260813T153038+0900_cv_378a4e89/selection.json")
P1_REFERENCE = Path("submissions/frozen/P1_FROZEN_READY_TO_UPLOAD_28243fda.csv")

P2_ROUTER_MODEL = Path("artifacts/p2_score_optimization_v1/model.joblib")
P2_DEEP_RESULT = Path("artifacts/p2_deep_finalists_v1/result.json")
P2_GATE_MODEL = Path("artifacts/p2_public_state_soft_gate_v1/gate_model.joblib")
P2_REFERENCE = Path("submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V2.csv")

P3_MODEL_DIR = Path("submissions/p3_frozen_catboost")
P3_ROUTER = Path("submissions/p3_lead_long_loss_router/router.joblib")
P3_REFERENCE = Path("submissions/p3_long_persistence_shrink/submission.csv")

STAGE_WEIGHTS = {
    "saved_P1": (0.00, 0.08, 90),
    "saved_P2": (0.08, 0.25, 600),
    "saved_P3": (0.25, 0.33, 120),
    "retrain_P1": (0.33, 0.44, 180),
    "retrain_P2": (0.44, 0.78, 1500),
    "retrain_P3": (0.78, 0.97, 900),
    "finalize": (0.97, 1.00, 60),
}

RAW_FILE_NAMES = {
    "train.csv",
    "test.csv",
    "observations.csv",
    "train_wave.csv",
    "train_atmos.csv",
    "test_context.parquet",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_rule.csv",
    "baseline_interp.csv",
    "baseline_persistence.csv",
    "score.py",
    "README.md",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)


class RunProgress:
    """Thread-safe, atomic status file suitable for a local PyCharm gauge."""

    def __init__(self, output_dir: Path, mode: str) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "status.json"
        self.log_path = output_dir / "logs" / "run.log"
        self.mode = mode
        self.started = time.time()
        self._lock = threading.Lock()
        self._fraction = 0.0
        self._stage = "initialize"
        self._detail = "출력 구조 준비"
        self._problem_status = {
            problem: {"saved_weight": "pending", "retrain": "pending"}
            for problem in ("P1", "P2", "P3")
        }
        self._last_log: tuple[str, str] | None = None
        if mode == "saved":
            for value in self._problem_status.values():
                value["retrain"] = "not_requested"
        elif mode == "retrain":
            for value in self._problem_status.values():
                value["saved_weight"] = "not_requested"
        self.write()

    def write(self, *, status: str = "running") -> None:
        with self._lock:
            elapsed = time.time() - self.started
            fraction = max(self._fraction, 0.001)
            remaining = elapsed * (1.0 - fraction) / fraction if fraction < 1 else 0.0
            now = datetime.now().astimezone()
            atomic_json(
                self.path,
                {
                    "title": "Ocean AI Data P1·P2·P3 일괄 재현",
                    "status": status,
                    "mode": self.mode,
                    "progress_percent": round(self._fraction * 100.0, 2),
                    "stage": self._stage,
                    "detail": self._detail,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta": (now + timedelta(seconds=max(remaining, 0))).isoformat(),
                    "problems": self._problem_status,
                    "updated_at": now.isoformat(),
                },
            )

    def update(self, fraction: float, stage: str, detail: str) -> None:
        with self._lock:
            self._fraction = max(self._fraction, min(max(fraction, 0.0), 1.0))
            self._stage = stage
            self._detail = detail
            current = (stage, detail)
            if current != self._last_log:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"{datetime.now().astimezone().isoformat()}\t{stage}\t{detail}\n")
                self._last_log = current
        self.write()

    def mark(self, problem: str, arm: str, status: str) -> None:
        with self._lock:
            self._problem_status[problem][arm] = status
        self.write()

    @contextlib.contextmanager
    def heartbeat(self, stage: str, detail: str):
        start, stop, expected = STAGE_WEIGHTS[stage]
        event = threading.Event()
        began = time.monotonic()

        def pulse() -> None:
            while not event.wait(5.0):
                elapsed = time.monotonic() - began
                local = min(elapsed / max(expected, 1), 0.92)
                self.update(start + (stop - start) * local, stage, detail)

        self.update(start, stage, detail)
        thread = threading.Thread(target=pulse, daemon=True)
        thread.start()
        try:
            yield lambda local, message=None: self.update(
                start + (stop - start) * min(max(float(local), 0.0), 1.0),
                stage,
                message or detail,
            )
        finally:
            event.set()
            thread.join(timeout=1)
            self.update(stop, stage, detail)

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            self._stage = "failed"
            self._detail = f"{type(exc).__name__}: {exc}"
        self.write(status="failed")

    def complete(self) -> None:
        with self._lock:
            self._fraction = 1.0
            self._stage = "complete"
            self._detail = "세 문제 재현·재학습·검증 완료, 업로드는 아직 수행하지 않음"
        self.write(status="complete")


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_stage_receipt(
    output_dir: Path, stage: str, result: dict[str, Any], artifacts: list[Path]
) -> Path:
    receipt = output_dir / "receipts" / ".stages" / f"{stage}.json"
    atomic_json(
        receipt,
        {
            "stage": stage,
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "result": result,
            "artifacts": [_artifact(path, output_dir) for path in artifacts],
        },
    )
    return receipt


def _resume_stage(output_dir: Path, stage: str) -> dict[str, Any] | None:
    receipt = output_dir / "receipts" / ".stages" / f"{stage}.json"
    if not receipt.is_file():
        return None
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        return None
    for artifact in value.get("artifacts", []):
        path = output_dir / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            return None
    return value["result"]


def _require_files(paths: list[Path]) -> None:
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen artifacts are missing: {missing}")


def _saved_p1(data_dir: Path, output: Path) -> dict[str, Any]:
    from p1_qc.config import load_config
    from p1_qc.data import load_dataset
    from p1_qc.pipeline import load_model, load_or_build_features, reproduce_submission

    _require_files([P1_MODEL, P1_CONFIG, P1_REFERENCE])
    config = load_config(P1_CONFIG)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    model = load_model(P1_MODEL)
    if config.features.mode != model.feature_mode:
        from dataclasses import replace

        config = replace(
            config,
            mode=model.feature_mode,
            features=replace(config.features, mode=model.feature_mode),
        )
    bundle = load_or_build_features(test, config, kind="test", use_cache=True)
    report = reproduce_submission(P1_MODEL, test, bundle, output, expected_path=P1_REFERENCE)
    report.pop("path", None)
    actual = sha256_file(output)
    if actual != EXPECTED_SHA["P1"]:
        raise RuntimeError(f"P1 byte SHA mismatch: {actual}")
    return {**report, "sha256": actual, "model_sha256": sha256_file(P1_MODEL)}


def _retrain_p1(data_dir: Path, target: Path) -> dict[str, Any]:
    from p1_qc.config import load_config
    from p1_qc.data import load_dataset
    from p1_qc.pipeline import (
        load_or_build_features,
        predict_submission,
        save_model,
        train_full_model,
    )
    from p1_qc.submission import validate_submission, write_submission

    config = load_config(P1_CONFIG)
    selection = json.loads(P1_SELECTION.read_text(encoding="utf-8"))
    train = load_dataset(data_dir / "train.csv", kind="train", audit=True)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    train_bundle = load_or_build_features(train, config, kind="train", use_cache=True)
    model = train_full_model(train, train_bundle, config, selection)
    model_path = target / "model.joblib"
    save_model(model, model_path)
    test_bundle = load_or_build_features(test, config, kind="test", use_cache=True)
    submission, _ = predict_submission(model, test, test_bundle)
    submission_path = write_submission(submission, target / "submission.csv")
    validation = validate_submission(submission_path, test)
    reference = pd.read_csv(P1_REFERENCE, keep_default_na=False)
    actual = pd.read_csv(submission_path, keep_default_na=False)
    mismatch = int((actual["label"].to_numpy() != reference["label"].to_numpy()).sum())
    return {
        **validation,
        "model_sha256": sha256_file(model_path),
        "submission_sha256": sha256_file(submission_path),
        "reference_label_mismatch_rows": mismatch,
        "reference_row_identical": actual.equals(reference),
    }


def _saved_p2(data_dir: Path, output: Path, update: Callable[[float, str], None]) -> dict[str, Any]:
    from p2_restore.final_inference import reproduce_final_submission

    _require_files([P2_ROUTER_MODEL, P2_DEEP_RESULT, P2_GATE_MODEL, P2_REFERENCE])

    def checkpoint(name: str, done: int, total: int) -> None:
        update(0.2 + 0.75 * done / max(total, 1), f"P2 checkpoint {done + 1}/{total}: {name}")

    return reproduce_final_submission(
        data_dir=data_dir,
        router_model_path=P2_ROUTER_MODEL,
        deep_result_path=P2_DEEP_RESULT,
        gate_model_path=P2_GATE_MODEL,
        output_path=output,
        expected_sha256=EXPECTED_SHA["P2"],
        progress=checkpoint,
    )


def _retrain_p2(
    data_dir: Path, target: Path, update: Callable[[float, str], None]
) -> dict[str, Any]:
    import gc

    import joblib
    import torch

    from p2_restore.data import load_p2_data
    from p2_restore.deep_data import build_panel
    from p2_restore.deep_training import TrainingConfig, train_full_model
    from p2_restore.features import build_test_features, build_training_features
    from p2_restore.final_inference import compose_final_prediction
    from p2_restore.research import (
        append_public_dynamics,
        append_public_m2_harmonics,
        select_lean_m2_dynamics,
    )
    from p2_restore.score_optimization import fit_score_router
    from p2_restore.submission import build_submission, validate_submission

    target.mkdir(parents=True, exist_ok=True)
    models_dir = target / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    data = load_p2_data(data_dir)
    frozen_router = joblib.load(P2_ROUTER_MODEL)
    update(0.03, "P2 router 학습 특징 생성")
    base = build_training_features(data.observations)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, data.observations)
    update(0.10, "P2 400-round router 전체 재학습")
    router = fit_score_router(base, lean, phase, dict(frozen_router.layer_arms))
    router_path = models_dir / "router.joblib"
    joblib.dump(router, router_path, compress=3)
    test_base = build_test_features(data)
    test_dynamic = append_public_dynamics(test_base, data.observations)
    test_lean = select_lean_m2_dynamics(test_base, test_dynamic)
    test_phase = append_public_m2_harmonics(test_lean, data.observations)
    router_prediction = router.predict_components(test_base, test_lean, test_phase)["router"]

    panel = build_panel(data.observations)
    deep_result = json.loads(P2_DEEP_RESULT.read_text(encoding="utf-8"))
    entries = [
        (name, entry) for name, current in deep_result["full_models"].items() for entry in current
    ]
    deep_predictions: dict[str, list[np.ndarray]] = {}
    model_receipts: list[dict[str, Any]] = []
    positions = panel.times.get_indexer(pd.to_datetime(data.test_index["time"], utc=True))
    layers = data.test_index["layer"].to_numpy(int) - 2
    if (positions < 0).any():
        raise ValueError("P2 retrain test times are absent from panel")
    for number, (name, entry) in enumerate(entries):
        payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        epochs = int(payload["epochs"])
        config = TrainingConfig(**payload["config"])

        current_seed = config.seed

        def epoch_progress(
            state: dict[str, object],
            n: int = number,
            model_name: str = name,
            model_seed: int = current_seed,
        ) -> None:
            local = float(state["epoch"]) / max(float(state["max_epochs"]), 1.0)
            update(
                0.18 + 0.72 * (n + local) / max(len(entries), 1),
                f"P2 {model_name} seed {model_seed}: epoch {state['epoch']}/{state['max_epochs']}",
            )

        trained = train_full_model(panel, config, epochs=epochs, progress=epoch_progress)
        checkpoint_path = models_dir / "deep" / f"{name}_seed{config.seed}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": name,
                "config": asdict(config),
                "epochs": epochs,
                "input_center": trained.normalizer.input_center,
                "input_scale": trained.normalizer.input_scale,
                "residual_center": trained.normalizer.residual_center,
                "residual_scale": trained.normalizer.residual_scale,
                "state_dict": trained.state_dict,
            },
            checkpoint_path,
        )
        deep_predictions.setdefault(name, []).append(trained.prediction[positions, layers])
        model_receipts.append(
            {
                "name": name,
                "seed": config.seed,
                "epochs": epochs,
                "sha256": sha256_file(checkpoint_path),
                "final_train_mse_c": trained.final_train_mse_c,
            }
        )
        del trained
        gc.collect()
        torch.cuda.empty_cache()
    averaged = {name: np.mean(values, axis=0) for name, values in deep_predictions.items()}
    gate = joblib.load(P2_GATE_MODEL)
    gate_path = models_dir / "gate_model.joblib"
    joblib.dump(gate, gate_path, compress=3)
    update(0.94, "P2 동결 stack·soft gate·물리 투영")
    prediction, diagnostics = compose_final_prediction(
        data,
        router_prediction=np.asarray(router_prediction),
        deep_predictions=averaged,
        stack_weights=deep_result["weights_by_layer"],
        gate_model=gate,
    )
    submission_path = target / "submission.csv"
    build_submission(data.test_index, prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(submission_path, data.test_index)
    reference = pd.read_csv(P2_REFERENCE)["temp"].to_numpy(float)
    difference = prediction - reference
    return {
        **validation,
        **diagnostics,
        "router_model_sha256": sha256_file(router_path),
        "gate_model_sha256": sha256_file(gate_path),
        "deep_models": model_receipts,
        "submission_sha256": sha256_file(submission_path),
        "reference_rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "reference_max_abs_difference": float(np.max(np.abs(difference))),
    }


def _saved_p3(data_dir: Path, output: Path) -> dict[str, Any]:
    from p3_wave.final_inference import reproduce_final_submission

    required = [
        P3_MODEL_DIR / "model.cbm",
        P3_MODEL_DIR / "model_multi.cbm",
        P3_MODEL_DIR / "feature_columns.json",
        P3_ROUTER,
        P3_REFERENCE,
    ]
    _require_files(required)
    return reproduce_final_submission(
        data_dir=data_dir,
        model_path=required[0],
        multi_model_path=required[1],
        feature_columns_path=required[2],
        router_path=P3_ROUTER,
        output_path=output,
        expected_sha256=EXPECTED_SHA["P3"],
    )


def _retrain_p3(
    data_dir: Path, target: Path, update: Callable[[float, str], None]
) -> dict[str, Any]:
    import joblib
    from catboost import CatBoostRegressor

    from p3_wave.data import LEADS, load_p3_data
    from p3_wave.features import build_test_features, build_training_features
    from p3_wave.final_inference import apply_saved_router, predict_catboost_components
    from p3_wave.models import compact_feature_columns, threshold_case_weights
    from p3_wave.persistence_shrink import apply_long_lead_persistence_shrink
    from p3_wave.submission import build_submission, validate_submission, write_submission
    from p3_wave.validation import expand_leads

    target.mkdir(parents=True, exist_ok=True)
    models = target / "models"
    features_dir = target / "features"
    models.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    data = load_p3_data(data_dir)
    update(0.03, "P3 20분 간격 전체 train 특징 재생성")

    def feature_progress(done: int, total: int) -> None:
        update(0.03 + 0.32 * done / max(total, 1), f"P3 특징 {done:,}/{total:,}")

    train_set = build_training_features(data, dense_spacing_minutes=20, progress=feature_progress)
    test_set = build_test_features(data)
    train_set.features.to_parquet(
        features_dir / "train_features.parquet", index=False, compression="zstd"
    )
    train_set.anchors.to_parquet(
        features_dir / "train_anchors.parquet", index=False, compression="zstd"
    )
    test_set.features.to_parquet(
        features_dir / "test_features.parquet", index=False, compression="zstd"
    )
    columns = compact_feature_columns(list(train_set.feature_columns))
    (models / "feature_columns.json").write_text(
        json.dumps(columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    anchors = train_set.anchors
    features = train_set.features
    train_ids = anchors["anchor_id"].to_numpy(np.int64)
    x_train, y_train, meta = expand_leads(features, anchors, train_ids, columns)

    update(0.38, "P3 single-output CatBoost 700 rounds")
    single = CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.2,
        random_seed=20260817,
        thread_count=8,
        verbose=False,
        allow_writing_files=False,
    )
    x_train = x_train.copy()
    x_train["station"] = x_train["station"].astype(str)
    x_train["lead_h"] = x_train["lead_h"].astype(str)
    single.fit(
        x_train,
        y_train,
        sample_weight=threshold_case_weights(meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single_path = models / "model.cbm"
    single.save_model(single_path)

    update(0.62, "P3 GPU multi-output CatBoost 1,200 rounds")
    lookup = features.set_index("anchor_id")
    multi_x = lookup.loc[train_ids, ["station", *columns]].reset_index(drop=True)
    multi_x["station"] = multi_x["station"].astype(str)
    anchor_lookup = anchors.set_index("anchor_id")
    multi_y = np.column_stack(
        [
            anchor_lookup.loc[train_ids, f"target_{lead}"].to_numpy(float)
            - anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
            for lead in LEADS
        ]
    )
    multi = CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=1200,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10.0,
        random_strength=0.15,
        random_seed=20260817,
        task_type="GPU",
        devices="0",
        boosting_type="Plain",
        verbose=False,
        allow_writing_files=False,
    )
    multi.fit(
        multi_x,
        multi_y,
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(float)
        ),
        cat_features=[0],
        verbose=False,
    )
    multi_path = models / "model_multi.cbm"
    multi.save_model(multi_path)
    router_copy = models / "router.joblib"
    shutil.copyfile(P3_ROUTER, router_copy)

    update(0.92, "P3 frozen router·long-lead persistence 합성")
    test_index = data.test_index[["case_id", "station", "lead_h"]].copy()
    single_pred, multi_pred, current = predict_catboost_components(
        test_set.features,
        test_index,
        model_path=single_path,
        multi_model_path=multi_path,
        feature_columns_path=models / "feature_columns.json",
    )
    routed = apply_saved_router(
        test_set.features,
        test_index,
        single_pred,
        multi_pred,
        current,
        joblib.load(router_copy),
    )
    persistence = pd.read_csv(data_dir / "baseline_persistence.csv")
    final = apply_long_lead_persistence_shrink(
        routed,
        persistence["hs_pred"].to_numpy(float),
        test_index["lead_h"].to_numpy(int),
    )
    submission_path = write_submission(
        build_submission(test_index, final), test_index, target / "submission.csv"
    )
    validate_submission(pd.read_csv(submission_path), test_index)
    reference = pd.read_csv(P3_REFERENCE)["hs_pred"].to_numpy(float)
    difference = final - reference
    return {
        "rows": len(final),
        "feature_count": len(columns),
        "model_sha256": sha256_file(single_path),
        "multi_model_sha256": sha256_file(multi_path),
        "router_sha256": sha256_file(router_copy),
        "submission_sha256": sha256_file(submission_path),
        "reference_rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "reference_max_abs_difference": float(np.max(np.abs(difference))),
        "gpu_retrain_is_nondeterministic": True,
    }


def _input_hashes(data_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    names = {
        "P1": ("train.csv", "test.csv", "sample_submission.csv", "baseline_rule.csv"),
        "P2": (
            "observations.csv",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_interp.csv",
        ),
        "P3": (
            "train_wave.csv",
            "train_atmos.csv",
            "test_context.parquet",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_persistence.csv",
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for problem, files in names.items():
        result[problem] = {}
        for name in files:
            path = data_dirs[problem] / name
            if not path.is_file():
                raise FileNotFoundError(f"{problem} input is missing {name}")
            result[problem][name] = sha256_file(path)
    return result


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "torch",
        "pyarrow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    git = subprocess.run(
        ["git", "status", "--short", "--branch"], capture_output=True, text=True, check=False
    )
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "packages": packages,
        "git_sha": sha.stdout.strip() if sha.returncode == 0 else "unknown",
        "git_dirty": bool([line for line in git.stdout.splitlines()[1:] if line.strip()]),
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else "unavailable",
    }


def _assert_no_raw_files(output_dir: Path) -> None:
    violations = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".zip" or path.name in RAW_FILE_NAMES:
            violations.append(path.relative_to(output_dir).as_posix())
    if violations:
        raise RuntimeError(f"raw/source-like files leaked into output: {violations}")


def _copy_ready(output_dir: Path, problem: str) -> Path:
    source = output_dir / "saved_weight" / f"{problem}_submission.csv"
    target = output_dir / "ready" / f"{problem}_submission.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != EXPECTED_SHA[problem]:
        raise RuntimeError(f"{problem} ready copy hash differs")
    return target


def _consolidate_receipts(
    output_dir: Path,
    mode: str,
    results: dict[str, dict[str, dict[str, Any]]],
) -> list[Path]:
    receipts = []
    for problem in ("P1", "P2", "P3"):
        value = {
            "problem": problem,
            "status": "ready_not_uploaded",
            "saved_weight": results.get("saved", {}).get(problem),
            "retrain": results.get("retrain", {}).get(problem),
            "expected_sha256": EXPECTED_SHA[problem],
            "uploaded": False,
        }
        if mode in {"saved", "both"}:
            value["ready"] = _artifact(
                output_dir / "ready" / f"{problem}_submission.csv", output_dir
            )
        path = output_dir / "receipts" / f"{problem}.json"
        atomic_json(path, value)
        receipts.append(path)
    return receipts


def _run_all_in_project(
    *,
    project_root: Path,
    p1_data_dir: Path,
    p2_data_dir: Path,
    p3_data_dir: Path,
    output_dir: Path,
    mode: str = "both",
    resume: bool = False,
) -> dict[str, Any]:
    if mode not in {"saved", "retrain", "both"}:
        raise ValueError("mode must be saved, retrain, or both")
    output_dir = output_dir.resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"output directory already exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ready", "saved_weight", "retrain", "receipts", "logs"):
        (output_dir / name).mkdir(exist_ok=True)
    progress = RunProgress(output_dir, mode)
    data_dirs = {
        "P1": p1_data_dir.resolve(),
        "P2": p2_data_dir.resolve(),
        "P3": p3_data_dir.resolve(),
    }
    results: dict[str, dict[str, dict[str, Any]]] = {"saved": {}, "retrain": {}}
    started = time.time()
    try:
        input_hashes = _input_hashes(data_dirs)
        environment_path = output_dir / "receipts" / "environment.json"
        atomic_json(environment_path, _environment())

        if mode in {"saved", "both"}:
            saved_functions: list[tuple[str, Callable[..., dict[str, Any]]]] = [
                ("P1", _saved_p1),
                ("P2", _saved_p2),
                ("P3", _saved_p3),
            ]
            for problem, function in saved_functions:
                stage = f"saved_{problem}"
                resumed = _resume_stage(output_dir, stage) if resume else None
                output = output_dir / "saved_weight" / f"{problem}_submission.csv"
                if resumed is not None:
                    results["saved"][problem] = resumed
                    progress.mark(problem, "saved_weight", "reused")
                    continue
                progress.mark(problem, "saved_weight", "running")
                with progress.heartbeat(stage, f"{problem} 저장 가중치 원본 재추론") as update:
                    if problem == "P2":
                        result = function(data_dirs[problem], output, update)
                    else:
                        result = function(data_dirs[problem], output)
                results["saved"][problem] = result
                _write_stage_receipt(output_dir, stage, result, [output])
                progress.mark(problem, "saved_weight", "complete")

        if mode in {"retrain", "both"}:
            retrain_functions: list[tuple[str, Callable[..., dict[str, Any]]]] = [
                ("P1", _retrain_p1),
                ("P2", _retrain_p2),
                ("P3", _retrain_p3),
            ]
            for problem, function in retrain_functions:
                stage = f"retrain_{problem}"
                resumed = _resume_stage(output_dir, stage) if resume else None
                target = output_dir / "retrain" / problem
                if resumed is not None:
                    results["retrain"][problem] = resumed
                    progress.mark(problem, "retrain", "reused")
                    continue
                progress.mark(problem, "retrain", "running")
                with progress.heartbeat(stage, f"{problem} 동결 사양 전체 재학습") as update:
                    if problem == "P1":
                        result = function(data_dirs[problem], target)
                    else:
                        result = function(data_dirs[problem], target, update)
                artifacts = list(target.rglob("*"))
                artifacts = [path for path in artifacts if path.is_file()]
                results["retrain"][problem] = result
                _write_stage_receipt(output_dir, stage, result, artifacts)
                progress.mark(problem, "retrain", "complete")

        with progress.heartbeat("finalize", "ready 복사·무결성·원본 미포함 최종 검사"):
            if mode in {"saved", "both"}:
                ready = [_copy_ready(output_dir, problem) for problem in ("P1", "P2", "P3")]
            else:
                ready = []
            _assert_no_raw_files(output_dir)
            receipt_paths = _consolidate_receipts(output_dir, mode, results)
            manifest = {
                "created_at": datetime.now().astimezone().isoformat(),
                "status": "complete_not_uploaded",
                "mode": mode,
                "elapsed_seconds": time.time() - started,
                "input_sha256": input_hashes,
                "expected_submission_sha256": EXPECTED_SHA,
                "ready": [_artifact(path, output_dir) for path in ready],
                "receipts": [_artifact(path, output_dir) for path in receipt_paths],
                "environment": _artifact(environment_path, output_dir),
                "source_data_copied": False,
                "external_observations_used": 0,
                "uploaded": False,
            }
            manifest_path = output_dir / "manifest.json"
            atomic_json(manifest_path, manifest)
            _assert_no_raw_files(output_dir)
        progress.complete()
        return manifest
    except BaseException as exc:
        progress.fail(exc)
        raise


def run_all(
    *,
    project_root: Path,
    p1_data_dir: Path,
    p2_data_dir: Path,
    p3_data_dir: Path,
    output_dir: Path,
    mode: str = "both",
    resume: bool = False,
) -> dict[str, Any]:
    """Run from the project root without leaking a process-wide cwd change."""

    previous = Path.cwd()
    try:
        os.chdir(project_root)
        return _run_all_in_project(
            project_root=project_root,
            p1_data_dir=p1_data_dir,
            p2_data_dir=p2_data_dir,
            p3_data_dir=p3_data_dir,
            output_dir=output_dir,
            mode=mode,
            resume=resume,
        )
    finally:
        os.chdir(previous)
